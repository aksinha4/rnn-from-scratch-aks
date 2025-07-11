from gate import AddGate, MultiplyGate
from activations import Tanh, Sigmoid
from output import Softmax
import numpy as np

mulGate = MultiplyGate()
addGate = AddGate()
tanhGate = Tanh()
sigmoidGate = Sigmoid()
softmaxGate = Softmax()

class GRULayer:
    def forward(self, x, prev_h, Ur, Uz, Ug, Wr, Wz, Wg, V, br, bz, bg, by):
        # x, (w, 1)
        # prev_h, (h, 1) 
        # Ur, (h, w) 
        # Uz, (h, w)
        # Ug, (h, w)
        # Wr, (h, h)
        # Wz, (h, h)
        # Wg, (h, h)
        # V, (w, h)
        # br, (h, 1)
        # bz, (h, 1)
        # bg, (h, 1)
        # by, (w, 1)

        self.mulur = mulGate.forward(Ur, x)                  # (h, w) . (w, 1) = (h, 1)
        self.mulwr = mulGate.forward(Wr, prev_h)             # (h, h) . (h, 1) = (h, 1)
        self.add1r = addGate.forward(self.mulwr, self.mulur) # (h, 1) + (h, 1) = (h, 1) 
        self.add2r = addGate.forward(self.add1r, br)         # (h, 1) + (h, 1) = (h, 1)
        self.r = sigmoidGate.forward(self.add2r)             # 1 / (1 - (h, 1)) = (h, 1)

        self.muluz = mulGate.forward(Uz, x)                   # (h, w) . (w, 1) = (h, 1)
        self.mulwz = mulGate.forward(Wz, prev_h)              # (h, h) . (h, 1) = (h, 1)
        self.add1z = addGate.forward(self.mulwz, self.muluz)  # (h, 1) + (h, 1) = (h, 1) 
        self.add2z = addGate.forward(self.add1z, bz)          # (h, 1) + (h, 1) = (h, 1)
        self.z = sigmoidGate.forward(self.add2z)              # 1 / (1 - (h, 1)) = (h, 1) 

        self.mulug = mulGate.forward(Ug, x)                   # (h, w) . (w, 1) = (h, 1)
        self.mulwg = mulGate.forward(Wg, self.r * prev_h)     # (h, h) . ((h, 1) * (h, 1) = (h, 1)
        self.add1g = addGate.forward(self.mulwg, self.mulug)  # (h, 1) + (h, 1) = (h, 1)
        self.add2g = addGate.forward(self.add1g, bg)          # (h, 1) + (h, 1) = (h, 1)
        self.g = tanhGate.forward(self.add2g)                 # np.tanh((h, 1)) = (h, 1)

        # Below is source of much confusion. Below equations are as per the 
        # original paper(https://arxiv.org/pdf/1406.1078v3) whereas in 
        # Colah's blog (https://colah.github.io/posts/2015-08-Understanding-LSTMs/), 
        # self.g and prev_h positions are swapped.
        self.mulg = (1 - self.z) * self.g                     # (1 - (h, 1)) * (h, 1) = (h, 1)
        self.mulh = self.z * prev_h                           # (h, 1) * (h, 1) = (h, 1)
        self.h = self.mulg + self.mulh                        # (h, 1) + (h, 1) = (h, 1)

        self.prev_h = prev_h                                  # (h, 1)

        self.mulv = mulGate.forward(V, self.h)                # (w, h) . (h, 1) = (w, 1)
        self.y = addGate.forward(self.mulv, by)               # (w, 1) + (w, 1) = (w, 1)

    def backward(self, x, prev_h, dnext_h, Ur, Uz, Ug, Wr, Wz, Wg, V, br, bz, bg, by, dldy):
        # https://medium.com/data-science/backpropagation-in-rnn-explained-bdf853b4e1c2
        # https://cran.r-project.org/web/packages/rnn/vignettes/GRU_units.html
        # http://practicalcryptography.com/miscellaneous/machine-learning/graphically-determining-backpropagation-equations/

        # x, (w, 1)
        # prev_h, (h, 1) 
        # Ur, (h, w) 
        # Uz, (h, w)
        # Ug, (h, w)
        # Wr, h x h
        # Wz, h x h
        # Wg, h x h
        # V, (w, h)
        # br, (h, 1)
        # bz, (h, 1)
        # bg, (h, 1)
        # by, (w, 1)

        dy, dby = addGate.backward(self.y, by, dldy) # (w,1) * 1(w,1), (w,1) * 1(w,1) = (w,1), (w,1)
        dV, dhv = mulGate.backward(V, self.h, dy) # (w,1) . (h,1).T, (w,h).T . (w,1) = (w,h), (h, 1)

        dh = dhv + dnext_h # (h,1) + (h,1) = (h,1)

        dz1a = dh * self.prev_h # (h,1) * (h,1) = (h,1)
        dnext_h1 = dh * self.z # (h,1) * (h,1) = (h,1)

        dz1b = 1 - (dh * self.g) # 1 - ((h,1) * (h,1)) = (h,1)
        dg1 = dh * (1 - self.z) # (h,1) * (1 - (h,1)) = (h,1)

        dz = dz1a + dz1b # (h,1) + (h,1) = (h,1)

        dg = tanhGate.backward(self.add2g, dg1) # (1 - np.square((h,1)) * (h,1) = (h,1)
        dadd1g, dbg = addGate.backward(self.add1g, bg, dg) # (h,1) * 1(h,1), (h,1) * 1(h,1) = (h,1), (h,1)
        dmulwg, dmulug = addGate.backward(self.mulwg, self.mulug, dadd1g) # (h,1) * 1(h,1), (h,1) * 1(h,1) = (h,1), (h,1)
        dWg, drnext_h = mulGate.backward(Wg, self.r * self.prev_h, dmulwg) # (h,1) . (h,1).T, (h,h).T . (h,1) = (h,h), (h, 1)
        dr = drnext_h * self.prev_h # (h,1) * (h,1) = (h,1)
        dnext_h2 = drnext_h * self.r # (h,1) * (h,1) = (h,1)
        dUg, _ = mulGate.backward(Ug, x, dmulug) # (h,1) . (w,1).T, (h,w).T . (h,1) = (h,w), (w, 1)
      
        dadd2z = sigmoidGate.backward(self.z, dz) # (1 - (h,1)) * (h,1) * (h,1) = (h,1)
        dadd1z, dbz = addGate.backward(self.add1z, bz, dadd2z) # (h,1) * 1(h,1), (h,1) * 1(h,1) = (h,1), (h,1)
        dmulwz, dmuluz = addGate.backward(self.mulwz, self.muluz, dadd1z) # (h,1) * 1(h,1), (h,1) * 1(h,1) = (h,1), (h,1)
        dWz, dnext_h3 = mulGate.backward(Wz, prev_h, dmulwz) # (h,1) . (h,1).T, (h,h).T . (h,1) = (h,h), (h, 1)
        dUz, _ = mulGate.backward(Uz, x, dmuluz) # (h,1) . (w,1).T, (h,w).T . (h,1) = (h,w), (w, 1)

        dadd2r = sigmoidGate.backward(self.r, dr) # (1 - (h,1)) * (h,1) * (h,1) = (h,1)
        dadd1r, dbr = addGate.backward(self.add1r, br, dadd2r) # (h,1) * 1(h,1), (h,1) * 1(h,1) = (h,1), (h,1)
        dmulwr, dmulur = addGate.backward(self.mulwr, self.mulur, dadd1r) # (h,1) * 1(h,1), (h,1) * 1(h,1) = (h,1), (h,1)
        dWr, dnext_h4 = mulGate.backward(Wr, prev_h, dmulwr) # (h,1) . (h,1).T, (h,h).T . (h,1) = (h,h), (h, 1)
        dUr, _ = mulGate.backward(Ur, x, dmulur) # (h,1) . (w,1).T, (h,w).T . (h,1) = (h,w), (w, 1)

        dnext_h = dnext_h1 + dnext_h2 + dnext_h3 + dnext_h4 # (h,1) + (h,1) + (h,1) + (h,1) = (h,1)

        return (dnext_h, dUr, dUz, dUg, dWr, dWz, dWg, dV, dbr, dbz, dbg, dby)
