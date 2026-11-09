"""Paper-derived PGSANet; implementation choices are listed in docs/IMPLEMENTATION.md."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def conv(cin, cout, kernel=3, dilation=1):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, padding=dilation*(kernel//2),
                                   dilation=dilation, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Residual(nn.Module):
    def __init__(self, cin, cout, dilation=1):
        super().__init__()
        self.body = nn.Sequential(conv(cin, cout, dilation=dilation),
                                  nn.Conv2d(cout, cout, 3, padding=dilation,
                                            dilation=dilation, bias=False), nn.BatchNorm2d(cout))
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return F.relu(self.body(x) + self.skip(x))


def summaries(x):
    return torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)


class SDCB(nn.Module):
    """Eqs. (1)-(7): FFT of the convolution KERNEL, not the input image."""
    def __init__(self, channels, bands=3, kernel=3):
        super().__init__()
        if kernel % 2 != 1 or bands < 1:
            raise ValueError('Require an odd kernel and positive band count')
        self.kernel = kernel
        self.weight = nn.Parameter(torch.empty(channels, channels, kernel, kernel))
        nn.init.kaiming_normal_(self.weight)
        self.mix = nn.Parameter(torch.eye(bands))
        self.noise3, self.noise5 = conv(channels, channels, 3), conv(channels, channels, 5)
        self.noise = nn.Conv2d(channels*3, 1, 1)
        self.attn = nn.Conv2d(2, bands, 1)
        self.lambdas = nn.Parameter(torch.ones(bands))
        self.residual_gate = nn.Conv2d(2, 1, 1)
        f = torch.fft.fftfreq(kernel)
        r = (f[:, None].square() + f[None, :].square()).sqrt()
        # Partition discrete radial levels; 3x3 has DC, axial and diagonal levels.
        levels = torch.unique(r, sorted=True)
        idx = torch.div(torch.searchsorted(levels, r.contiguous()) * bands,
                        len(levels), rounding_mode='floor').clamp(max=bands-1)
        masks = torch.stack([(idx == b).float() for b in range(bands)])
        if not masks.flatten(1).any(1).all():
            raise ValueError('Empty frequency band; increase kernel size or reduce bands')
        self.register_buffer('masks', masks)

    def forward(self, x):
        spectrum = torch.fft.fft2(self.weight.float())
        fused_masks = torch.einsum('ab,bij->aij', self.mix, self.masks)
        kernels = torch.fft.ifft2(fused_masks[:, None, None] * spectrum).real
        context = x.mean((2, 3), keepdim=True).expand_as(x)
        noise = torch.sigmoid(self.noise(torch.cat([self.noise3(x), self.noise5(x), context], 1)))
        attention = torch.sigmoid(self.attn(summaries(x)) - self.lambdas[None, :, None, None]*noise)
        y = sum(attention[:, b:b+1] * F.conv2d(x, w.to(x.dtype), padding=self.kernel//2)
                for b, w in enumerate(kernels))
        gate = torch.sigmoid(self.residual_gate(summaries(x)))
        return gate*y + (1-gate)*x


def sobel(x):
    gx = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None]
    c = x.shape[1]
    return (F.conv2d(x, gx.expand(c, 1, 3, 3), padding=1, groups=c),
            F.conv2d(x, gx.transpose(-1, -2).expand(c, 1, 3, 3), padding=1, groups=c))


class ABDA(nn.Module):
    """Eqs. (8)-(17), bilinear deformable sampling with grid_sample."""
    def __init__(self, channels, windows=(3, 5)):
        super().__init__()
        if any(w % 2 != 1 or w < 1 for w in windows):
            raise ValueError('Sampling windows must be positive odd numbers')
        self.windows = windows
        self.confidence = nn.Conv2d(1, 1, 3, padding=1)
        self.offsets = nn.ModuleList([nn.Conv2d(channels+2, 2*w*w, 3, padding=1) for w in windows])
        self.gamma, self.beta = nn.Parameter(torch.tensor(1.)), nn.Parameter(torch.tensor(1.))
        self.fuse = conv(channels*len(windows), channels, 1)

    def forward(self, x):
        n, c, h, w = x.shape
        dx, dy = sobel(x)
        boundary = (dx.abs()+dy.abs()).sum(1, keepdim=True)
        confidence = torch.sigmoid(self.confidence(x.mean(1, keepdim=True)))
        cues = torch.cat([x, boundary, confidence], 1)
        yy, xx = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype),
                                torch.arange(w, device=x.device, dtype=x.dtype), indexing='ij')
        base = torch.stack([xx, yy], -1)[None]
        outputs = []
        for size, predict in zip(self.windows, self.offsets):
            offsets = predict(cues).reshape(n, size*size, 2, h, w)
            values, energies = [], []
            for k in range(size*size):
                displacement = x.new_tensor([k % size-size//2, k//size-size//2])
                grid = base + displacement + offsets[:, k].permute(0, 2, 3, 1)
                grid = torch.stack([2*(grid[..., 0]+.5)/w-1, 2*(grid[..., 1]+.5)/h-1], -1)
                sampled = F.grid_sample(torch.cat([x, boundary, confidence], 1), grid,
                                        mode='bilinear', padding_mode='border', align_corners=False)
                values.append(sampled[:, :c])
                energies.append((x*sampled[:, :c]).sum(1) + self.gamma*sampled[:, c] + self.beta*sampled[:, c+1])
            weights = torch.stack(energies, 1).softmax(1)
            outputs.append((torch.stack(values, 1)*weights[:, :, None]).sum(1))
        return self.fuse(torch.cat(outputs, 1))


class PAFE(nn.Module):
    """Eqs. (18)-(29). Exact cross-attention, chunked over queries to limit memory."""
    def __init__(self, channels, prior_channels=3, experts=3, attention_grid=16, chunk=256):
        super().__init__()
        self.prior = conv(prior_channels, channels, 1)
        self.phys = nn.Conv2d(channels, 1, 1)
        self.gamma = nn.Parameter(torch.ones(3))
        self.attention = nn.Conv2d(1, 1, 7, padding=3)
        self.route = nn.Conv2d(2*channels, experts, 1)
        self.experts = nn.ModuleList([Residual(2*channels, channels, dilation=i+1) for i in range(experts)])
        self.q, self.k, self.v, self.proj = [nn.Conv2d(channels, channels, 1) for _ in range(4)]
        self.alpha = nn.Parameter(torch.tensor(1.))
        self.attention_grid, self.chunk = attention_grid, chunk

    def forward(self, x, prior):
        prior = F.interpolate(prior, size=x.shape[-2:], mode='bilinear', align_corners=False)
        p = self.prior(prior)
        cat = torch.cat([x, p], 1)
        t = self.gamma[0]*self.phys(p) + self.gamma[1]*cat.mean(1, keepdim=True) + self.gamma[2]*cat.amax(1, keepdim=True)
        attended = cat * torch.sigmoid(self.attention(t))
        routing = self.route(attended).softmax(1)
        out = sum(routing[:, i:i+1]*expert(attended) for i, expert in enumerate(self.experts))
        # Bounded key/value grid is an explicitly documented engineering default.
        if self.attention_grid:
            p = F.adaptive_avg_pool2d(p, (min(p.shape[-2], self.attention_grid), min(p.shape[-1], self.attention_grid)))
        q = self.q(x).flatten(2).transpose(1, 2)
        k, v = self.k(p).flatten(2), self.v(p).flatten(2).transpose(1, 2)
        cross = torch.cat([(torch.bmm(qpart, k)/math.sqrt(x.shape[1])).softmax(-1).bmm(v)
                           for qpart in q.split(self.chunk, dim=1)], 1)
        cross = cross.transpose(1, 2).reshape_as(x)
        return out + self.alpha*self.proj(cross)


class PGSANet(nn.Module):
    def __init__(self, num_classes, in_channels=1, widths=(32, 64, 128, 256), bands=3,
                 windows=(3, 5), experts=3, prior_channels=3, dropout=.5,
                 attention_grid=16, use_sdcb=True, use_abda=True, use_pafe=True):
        super().__init__()
        self.use_pafe = use_pafe
        self.encoder = nn.ModuleList()
        previous = in_channels
        for width in widths:
            self.encoder.append(nn.Sequential(Residual(previous, width), SDCB(width, bands) if use_sdcb else nn.Identity(),
                                               nn.BatchNorm2d(width), nn.ReLU()))
            previous = width
        self.abda = ABDA(widths[-1], windows) if use_abda else nn.Identity()
        self.up, self.decoder = nn.ModuleList(), nn.ModuleList()
        for cin, cout in zip(widths[:0:-1], widths[-2::-1]):
            self.up.append(nn.ConvTranspose2d(cin, cout, 2, stride=2))
            self.decoder.append(nn.Sequential(Residual(2*cout, cout), nn.Dropout2d(dropout)))
        c = widths[0]
        self.pafe = PAFE(c, prior_channels, experts, attention_grid) if use_pafe else None
        self.aspp = nn.ModuleList([conv(c, c, 1)] + [conv(c, c, dilation=d) for d in (6, 12, 18)])
        self.pool_proj = nn.Sequential(nn.Conv2d(c, c, 1), nn.ReLU())
        self.head = nn.Sequential(conv(c*5, c, 1), nn.Conv2d(c, num_classes, 1))
        self.physics_alignment = nn.Conv2d(prior_channels, num_classes, 1)
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if isinstance(self.abda, ABDA):
            for module in self.abda.offsets:
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, image, prior=None):
        if self.use_pafe and prior is None:
            raise ValueError('PAFE requires a physical prior; supply it explicitly')
        skip, x = [], image
        for i, stage in enumerate(self.encoder):
            if i:
                x = F.max_pool2d(x, 2)
            x = stage(x)
            skip.append(x)
        x = self.abda(x)
        for up, decode, feature in zip(self.up, self.decoder, reversed(skip[:-1])):
            x = up(x)
            x = F.interpolate(x, size=feature.shape[-2:], mode='bilinear', align_corners=False)
            x = decode(torch.cat([x, feature], 1))
        if self.pafe is not None:
            x = self.pafe(x, prior)
        pooled = self.pool_proj(F.adaptive_avg_pool2d(x, 1))
        pooled = F.interpolate(pooled, size=x.shape[-2:], mode='bilinear', align_corners=False)
        logits = self.head(torch.cat([branch(x) for branch in self.aspp] + [pooled], 1))
        aligned = None if prior is None else self.physics_alignment(F.interpolate(prior, size=image.shape[-2:], mode='bilinear', align_corners=False)).softmax(1)
        return {'logits': logits, 'physics': aligned}
