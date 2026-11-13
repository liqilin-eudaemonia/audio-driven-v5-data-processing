"""Verified candidates for removing frozen work; no source-file modification."""
import importlib
from types import MethodType


def install(model, *, pad=True, weights=False, no_feats=False):
    import torch
    import torch.nn.functional as f
    encoder = importlib.import_module('float.models.encoder')
    original_pad = encoder.upfirdn2d_native
    restorers = []
    keepalive = []

    if pad:
        def fast_pad(input, kernel, up_x, up_y, down_x, down_y,
                     pad_x0, pad_x1, pad_y0, pad_y1):
            if (up_x, up_y, down_x, down_y) != (1, 1, 1, 1) or min(pad_x0, pad_x1, pad_y0, pad_y1) < 0:
                return original_pad(input, kernel, up_x, up_y, down_x, down_y,
                                    pad_x0, pad_x1, pad_y0, pad_y1)
            _, minor, height, width = input.shape
            kh, kw = kernel.shape
            # The native implementation first pads with six zeros (a clone),
            # then adds its real border. Combine those exact copy operations.
            out = input.view(-1, minor, height, 1, width, 1)
            out = f.pad(out, [0, 0, pad_x0, pad_x1, 0, 0, pad_y0, pad_y1])
            out = out.reshape(-1, 1, height + pad_y0 + pad_y1,
                              width + pad_x0 + pad_x1)
            out = f.conv2d(out, torch.flip(kernel, [0, 1]).view(1, 1, kh, kw))
            return out.reshape(-1, minor,
                               height + pad_y0 + pad_y1 - kh + 1,
                               width + pad_x0 + pad_x1 - kw + 1)
        encoder.upfirdn2d_native = fast_pad
        restorers.append(lambda: setattr(encoder, 'upfirdn2d_native', original_pad))

    if weights:
        for module in model.enc.modules():
            if isinstance(module, encoder.EqualConv2d):
                old = module.forward
                weight = (module.weight * module.scale).detach()
                keepalive.append(weight)
                def forward(self, x, cached=weight):
                    return f.conv2d(x, cached, bias=self.bias, stride=self.stride, padding=self.padding)
                module.forward = MethodType(forward, module)
                restorers.append(lambda m=module, fn=old: setattr(m, 'forward', fn))
            elif isinstance(module, encoder.EqualLinear):
                old = module.forward
                weight = (module.weight * module.scale).detach()
                bias = (module.bias * module.lr_mul).detach() if module.bias is not None else None
                keepalive.extend([weight, bias])
                def forward(self, x, cached=weight, cb=bias):
                    if self.activation:
                        return encoder.fused_leaky_relu(f.linear(x, cached), cb)
                    return f.linear(x, cached, bias=cb)
                module.forward = MethodType(forward, module)
                restorers.append(lambda m=module, fn=old: setattr(m, 'forward', fn))

    if no_feats:
        app = model.enc.net_app
        old = app.forward
        def forward(self, x):
            for conv in self.convs:
                x = conv(x)
            return x.squeeze(-1).squeeze(-1), []
        app.forward = MethodType(forward, app)
        restorers.append(lambda: setattr(app, 'forward', old))

    def restore():
        for fn in reversed(restorers):
            fn()
    restore.keepalive = keepalive
    return restore
