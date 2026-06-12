import torch
import torch.nn as nn
from analysis.IQA_optimization.SteerPyrUtils import *  


class SteerablePyramid(nn.Module):
    # refer to https://github.com/LabForComputationalVision/pyrtools
    # https://github.com/olivierhenaff/steerablePyramid
    def __init__(self, imgSize=[256,256], K=4, N=4, hilb=True, includeHF=True, device=torch.device("cuda")):
        super().__init__()
        assert imgSize[0]==imgSize[1]
        size = [ imgSize[0], imgSize[1]//2 + 1 ]
        # self.imgSize = imgSize
        self.hl0 = HL0_matrix( size ).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)

        self.l = []
        self.b = []
        self.s = []

        self.K    = K 
        self.N    = N 
        self.hilb = hilb
        self.includeHF = includeHF 

        self.indF = [ freq_shift( size[0], True, device  ) ] 
        self.indB = [ freq_shift( size[0], False, device ) ] 


        for n in range( self.N ):

            l = L_matrix_cropped( size ).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
            b = B_matrix(      K, size ).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
            s = S_matrix(      K, size ).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)

            self.l.append( l.div_(4) )
            self.b.append( b )
            self.s.append( s )

            size = [ l.size(-2), l.size(-1) ]

            self.indF.append( freq_shift( size[0], True, device ) )
            self.indB.append( freq_shift( size[0], False, device ) )


    def forward(self, x):
        # rfft2 returns a complex tensor of shape (..., H, W//2+1)
        fftfull = torch.fft.rfft2(x)                          # complex, no trailing dim
        xreal = fftfull.real
        xim   = fftfull.imag
        x = torch.cat((xreal.unsqueeze(1), xim.unsqueeze(1)), 1).unsqueeze(-3)
        x = torch.index_select(x, -2, self.indF[0])

        x   = self.hl0 * x
        h0f = x.select(-3, 0).unsqueeze(-3)
        l0f = x.select(-3, 1).unsqueeze(-3)
        lf  = l0f

        output = []

        for n in range(self.N):
            bf = self.b[n] * lf
            lf = self.l[n] * central_crop(lf)
            if self.hilb:
                hbf = self.s[n] * torch.cat((bf.narrow(1,1,1), -bf.narrow(1,0,1)), 1)
                bf  = torch.cat((bf, hbf), -3)
            if self.includeHF and n == 0:
                bf  = torch.cat((h0f, bf), -3)
            output.append(bf)

        output.append(lf)

        for n in range(len(output)):
            output[n] = torch.index_select(output[n], -2, self.indB[n])
            sig_size  = [output[n].shape[-2], (output[n].shape[-1]-1)*2]
            # reconstruct complex tensor for irfft2
            cplx = torch.complex(output[n].select(1,0), output[n].select(1,1))
            output[n] = torch.fft.irfft2(cplx, s=sig_size)

        if self.includeHF:
            output.insert(0, output[0].narrow(-3, 0, 1))
            output[1] = output[1].narrow(-3, 1, output[1].size(-3)-1)

        for n in range(len(output)):
            if self.hilb:
                if ((not self.includeHF) or 0 < n) and n < len(output)-1:
                    nfeat = output[n].size(-3)//2
                    o1 =  output[n].narrow(-3,     0, nfeat).unsqueeze(1)
                    o2 = -output[n].narrow(-3, nfeat, nfeat).unsqueeze(1)
                    output[n] = torch.cat((o2, o1), 1)
                else:
                    output[n] = output[n].unsqueeze(1)

        for n in range(len(output)):
            if n > 0:
                output[n] = output[n] * (2 ** (n-1))

        return output