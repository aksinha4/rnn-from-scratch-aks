from gate import AddGate, MultiplyGate
from activations import Tanh
from output import Softmax

mulGate = MultiplyGate()
addGate = AddGate()
tanhGate = Tanh()

class RNNLayer:
    def forward(self, x, prev_h, U, W, V, bh, by):
        # x, (w, 1)
        # prev_h, (h, 1) 
        # U, (h, w) 
        # W, (h, h
        # V, (w, h)
        # bh, (h, 1)
        # by, (w, 1)

        self.mulu = mulGate.forward(U, x) # (h, w) . (w, 1) = (h, 1)
        self.mulw = mulGate.forward(W, prev_h) # (h, h) . (h, 1) = (h, 1)
        self.add1 = addGate.forward(self.mulw, self.mulu) # (h, 1) + (h, 1) = (h, 1)
        self.add2 = addGate.forward(self.add1, bh) # (h, 1) . (h, 1) = (h, 1)
        self.h = tanhGate.forward(self.add2)  # np.tanh((h, 1)) = (h, 1)
        self.prev_h = prev_h # (h, 1)
        self.mulv = mulGate.forward(V, self.h) # (w, h) . (h, 1) = (w, 1)
        self.z = addGate.forward(self.mulv, by) # (w, 1) + (w, 1) = (w, 1)

    def backward(self, x, prev_h, dnext_h, U, W, V, bh, by, dldy):
        # https://medium.com/data-science/backpropagation-in-rnn-explained-bdf853b4e1c2
        # https://cran.r-project.org/web/packages/rnn/vignettes/GRU_units.html
        # http://practicalcryptography.com/miscellaneous/machine-learning/graphically-determining-backpropagation-equations/

        # x, (w, 1)
        # prev_h, (h, 1) 
        # dnext_h, (h, 1) 
        # U, (h, w) 
        # W, (h, h
        # V, (w, h)
        # bh, (h, 1)
        # by, (w, 1)
        # dldy, (w, 1)

        dmulv, dby = addGate.backward(self.z, by, dldy) # (w,1) * 1(w,1), (w,1) * 1(w,1) = (w,1), (w,1)
        dV, dhv = mulGate.backward(V, self.h, dmulv) # (w,1) . (h,1).T, (w,h).T . (w,1) = (w,h), (h, 1)
        dh = dhv + dnext_h # (h,1)
        ds = tanhGate.backward(self.h, dh) # (1 - np.square((h,1)) * (h,1) = (h,1)
        dadd1, dbh = addGate.backward(self.add2, bh, ds) # (h,1) * 1(h,1), (h,1) * 1(h,1) = (h,1), (h,1)
        dmulw, dmulu = addGate.backward(self.mulw, self.mulu, dadd1)  # (h,1) * 1(h,1), (h,1) * 1(h,1) = (h,1), (h,1)
        dW, dnext_h = mulGate.backward(W, prev_h, dmulw) # (h,1) . (h,1).T, (h,h).T . (h,1) = (h,h), (h, 1)
        dU, _ = mulGate.backward(U, x, dmulu)# (h,1) . (w,1).T, (h,w).T . (h,1) = (h,w), (w, 1)
        return (dnext_h, dU, dW, dV, dbh, dby) # (h, 1), (h, w), (h, h), (w, h), (h, 1), (w, 1)
