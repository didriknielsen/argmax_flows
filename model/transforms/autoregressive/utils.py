import torch


class InvertRasterScan():
    '''
    Invert autoregressive bijection in
    raster scan order (pixelwise, width first, then height).
    Data is assumed to be images of shape (C, H, W).
    Args:
        order (str): The order in which to invert. Choices: `{'cwh', 'wh'}`.
    '''

    def __init__(self, order='cwh'):
        assert order in {'cwh', 'wh'}
        self.order = order
        self.ready = False

    def setup(self, ar_net, element_inverse_fn):
        self.ar_net = ar_net
        self.element_inverse_fn = element_inverse_fn
        self.ready = True

    def inverse(self, z, **kwargs):
        assert self.ready, 'Run scheme.setup(...) before scheme.inverse(...).'
        with torch.no_grad():
            if self.order == 'cwh': x = self._inverse_cwh(z, **kwargs)
            if self.order == 'wh': x = self._inverse_wh(z, **kwargs)
        return x

    def _inverse_cwh(self, z, **kwargs):
        _, C, H, W = z.shape
        x = torch.zeros_like(z)
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    element_params = self.ar_net(x, **kwargs)
                    x[:,c,h,w] = self.element_inverse_fn(z[:,c,h,w], element_params[:,c,h,w])
        return x

    def _inverse_wh(self, z, **kwargs):
        _, C, H, W = z.shape
        x = torch.zeros_like(z)
        for h in range(H):
            for w in range(W):
                    element_params = self.ar_net(x, **kwargs)
                    x[:,:,h,w] = self.element_inverse_fn(z[:,:,h,w], element_params[:,:,h,w])
        return x


class InvertSequentialCL():
    '''
    Invert autoregressive bijection in sequential order.
    Data is assumed to be audio / time series of shape (C, L).
    Args:
        shape (Iterable): The data shape, e.g. (2,1024).
        order (str): The order in which to invert. Choices: `{'cl', 'l'}`.
    '''

    def __init__(self, order='cl'):
        assert order in {'cl', 'l'}
        self.order = order
        self.ready = False

    def setup(self, ar_net, element_inverse_fn):
        self.ar_net = ar_net
        self.element_inverse_fn = element_inverse_fn
        self.ready = True

    def inverse(self, z, **kwargs):
        assert self.ready, 'Run scheme.setup(...) before scheme.invert(...).'
        with torch.no_grad():
            if self.order == 'cl': x = self._inverse_cl(z, **kwargs)
            if self.order == 'l': x = self._inverse_l(z, **kwargs)
        return x

    def _inverse_cl(self, z, **kwargs):
        _, C, L = z.shape
        x = torch.zeros_like(z)
        for l in range(L):
            for c in range(C):
                element_params = self.ar_net(x, **kwargs)
                x[:,c,l] = self.element_inverse_fn(z[:,c,l], element_params[:,c,l])
        return x

    def _inverse_l(self, z, **kwargs):
        _, C, L = z.shape
        x = torch.zeros_like(z)
        for l in range(L):
            element_params = self.ar_net(x, **kwargs)
            x[:,:,l] = self.element_inverse_fn(z[:,:,l], element_params[:,:,l])
        return x


class InvertSequential():
    '''
    Invert autoregressive bijection in sequential order.
    Data is assumed to be time series of shape (L,).
    Args:
        shape (Iterable): The data shape, e.g. (1024,).
        order (str): The order in which to invert. Choices: `{'l'}`.
    '''

    def __init__(self, order='l'):
        assert order in {'l'}
        self.order = order
        self.ready = False

    def setup(self, ar_net, element_inverse_fn):
        self.ar_net = ar_net
        self.element_inverse_fn = element_inverse_fn
        self.ready = True

    def inverse(self, z, **kwargs):
        assert self.ready, 'Run scheme.setup(...) before scheme.inverse(...).'
        with torch.no_grad():
            _, L = self.shape
            x = torch.zeros_like(z)
            for l in range(L):
                element_params = self.ar_net(x, **kwargs)
                x[:,l] = self.element_inverse_fn(z[:,l], element_params[:,l])
        return x
