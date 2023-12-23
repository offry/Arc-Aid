import torch

class CompactBilinearPooling(torch.nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim, sum_pool = True):
        super(CompactBilinearPooling, self).__init__()
        self.output_dim = output_dim
        self.sum_pool = sum_pool
        generate_sketch_matrix = lambda rand_h, rand_s, input_dim, output_dim: torch.sparse.FloatTensor(torch.stack([torch.arange(input_dim, out = torch.LongTensor()), rand_h.long()]), rand_s.float(), [input_dim, output_dim]).to_dense()
        self.sketch_matrix1 = torch.nn.Parameter(generate_sketch_matrix(torch.randint(output_dim, size = (input_dim1,)), 2 * torch.randint(2, size = (input_dim1,)) - 1, input_dim1, output_dim))
        self.sketch_matrix2 = torch.nn.Parameter(generate_sketch_matrix(torch.randint(output_dim, size = (input_dim2,)), 2 * torch.randint(2, size = (input_dim2,)) - 1, input_dim2, output_dim))

    def forward(self, x1, x2):

        fft1 = torch.rfft(x1.permute(0, 2, 3, 1).matmul(self.sketch_matrix1), 1)
        fft2 = torch.rfft(x2.permute(0, 2, 3, 1).matmul(self.sketch_matrix2), 1)

        fft_product = torch.stack([fft1[..., 0] * fft2[..., 0] - fft1[..., 1] * fft2[..., 1], fft1[..., 0] * fft2[..., 1] + fft1[..., 1] * fft2[..., 0]], dim = -1)
        cbp = torch.irfft(fft_product, 1, signal_sizes = (self.output_dim,)) * self.output_dim

        return cbp.sum(dim = 1).sum(dim = 1) if self.sum_pool else cbp.permute(0, 3, 1, 2)
