from gate import AddGate, MultiplyGate
from activations import Tanh, Sigmoid
from output import Softmax
import numpy as np

mulGate = MultiplyGate()
addGate = AddGate()
tanhGate = Tanh()
sigmoidGate = Sigmoid()
softmaxGate = Softmax() 

class LSTMLayer:
    def forward(self, x, prev_h, prev_c, Ui, Uf, Uo, Ug, Wi, Wf, Wo, Wg, V, bi, bf, bo, bg, by):
        # x, (w, 1)
        # prev_h, (h, 1) 
        # prev_c, (h, 1) 
        # Ui, (h, w) 
        # Uf, (h, w)
        # Uo, (h, w)
        # Ug, (h, w)
        # Wi, (h, h)
        # Wf, (h, h)
        # Wo, (h, h)
        # Wg, (h, h)
        # V, (w, h)
        # bi, (h, 1)
        # bf, (h, 1)
        # bo, (h, 1)
        # bg, (h, 1)
        # by, (w, 1)

        self.mului = mulGate.forward(Ui, x) # (h, w) . (w, 1) = (h, 1)
        self.mulwi = mulGate.forward(Wi, prev_h) # (h, h) . (h, 1) = (h, 1)
        self.add1i = addGate.forward(self.mulwi, self.mului) # (h, 1) + (h, 1) = (h, 1)
        self.add2i = addGate.forward(self.add1i, bi) # (h, 1) + (h, 1) = (h, 1)
        self.i = sigmoidGate.forward(self.add2i) # 1 / (1 - (h, 1)) = (h, 1)

        self.muluf = mulGate.forward(Uf, x) # (h, w) . (w, 1) = (h, 1)
        self.mulwf = mulGate.forward(Wf, prev_h) # (h, h) . (h, 1) = (h, 1)
        self.add1f = addGate.forward(self.mulwf, self.muluf) # (h, 1) + (h, 1) = (h, 1)
        self.add2f = addGate.forward(self.add1f, bf) # (h, 1) + (h, 1) = (h, 1)
        self.f = sigmoidGate.forward(self.add2f)  # 1 / (1 - (h, 1)) = (h, 1)

        self.muluo = mulGate.forward(Uo, x)  # (h, w) . (w, 1) = (h, 1)
        self.mulwo = mulGate.forward(Wo, prev_h)  # (h, h) . (h, 1) = (h, 1)
        self.add1o = addGate.forward(self.mulwo, self.muluo) # (h, 1) + (h, 1) = (h, 1)
        self.add2o = addGate.forward(self.add1o, bo) # (h, 1) + (h, 1) = (h, 1)
        self.o = sigmoidGate.forward(self.add2o) # 1 / (1 - (h, 1)) = (h, 1)

        self.mulug = mulGate.forward(Ug, x) # (h, w) . (w, 1) = (h, 1)
        self.mulwg = mulGate.forward(Wg, prev_h) # (h, h) . (h, 1) = (h, 1)
        self.add1g = addGate.forward(self.mulwg, self.mulug) # (h, 1) + (h, 1) = (h, 1)
        self.add2g = addGate.forward(self.add1g, bg) # (h, 1) + (h, 1) = (h, 1)
        self.g = tanhGate.forward(self.add2g) # np.tanh((h, 1)) = (h, 1)

        self.prev_c = prev_c # (h,1)

        self.mulcprev = self.f * self.prev_c # (h,1) * (h,1) = (h,1)
        self.mulg = self.i * self.g # (h,1) * (h,1) = (h,1)
        self.c = self.mulcprev + self.mulg # (h,1) + (h,1) = (h,1)

        self.htemp = tanhGate.forward(self.c) # np.tanh(h,1) = (h,1)
        self.h = self.o * self.htemp # (h,1) + (h,1) = (h,1)

        self.prev_h = prev_h # (h,1)

        self.mulv = mulGate.forward(V, self.h) # (w, h) . (h, 1) = (w, 1)
        self.z = addGate.forward(self.mulv, by) # (w, 1) + (w, 1) = (w, 1)


    def backward(self, x, prev_h, prev_c, dnext_h, dnext_c, Ui, Uf, Uo, Ug, Wi, Wf, Wo, Wg, V, bi, bf, bo, bg, by, dldy):
        # https://medium.com/data-science/backpropagation-in-rnn-explained-bdf853b4e1c2
        # https://cran.r-project.org/web/packages/rnn/vignettes/GRU_units.html
        # http://practicalcryptography.com/miscellaneous/machine-learning/graphically-determining-backpropagation-equations/

        # x, (w, 1)
        # prev_h, (h, 1) 
        # prev_c, (h, 1) 
        # Ui, (h, w) 
        # Uf, (h, w)
        # Uo, (h, w)
        # Ug, (h, w)
        # Wi, (h, h)
        # Wf, (h, h)
        # Wo, (h, h) 
        # Wg, (h, h)
        # V, (w, h)
        # bi, (h, 1)
        # bf, (h, 1)
        # bo, (h, 1)
        # bg, (h, 1)
        # by, (w, 1)
        # dldy, (w, 1)

        """Backpropagation through a computational graph is a way to calculate gradients for complex neural network architectures like LSTMs. It involves traversing the graph, which represents the network's operations, backward to compute how much each weight and bias contributed to the overall error. This process allows the network to adjust its parameters and learn from data. [1, 2, 3, 4, 5, 6, 7, 8]  
            Here's a breakdown of the process: 
            1. Computational Graph Representation: 

            • A computational graph visually represents the flow of data and operations within a neural 
              network. [4, 9, 10]  
            • Nodes in the graph represent operations (e.g., matrix multiplication, addition, activation 
              functions), and edges represent the flow of data between these operations. [4, 4, 9, 9]  
            • For LSTMs, the graph includes nodes for each gate (input, forget, output, and cell state) 
              and the recurrent connections. [11, 12, 13, 14]  

            2. Backpropagation in a Nutshell: 

            • Backpropagation is the process of computing gradients (derivatives) of the loss function 
              with respect to each parameter in the network. [2]  
            • It uses the chain rule of calculus to efficiently calculate these gradients, starting from 
            the output and working backward through the graph. [2, 15]  

            3. LSTM-Specific Considerations: 

            • Gates: LSTMs have multiple gates (input, forget, output, and cell state) that regulate the 
              flow of information. [11, 11, 12, 12, 16, 17, 18, 19]  
            • Cell State: The cell state acts as a memory unit, preserving information over time. 
              [11, 11, 12, 12, 20, 21, 22]  
            • Backpropagation Through Time (BPTT): For sequential data, BPTT unfolds the LSTM network 
              through time, treating each time step as a layer. The gradients then flow backward through 
              this unfolded graph, considering the dependencies between time steps. [5, 5, 23, 23, 24, 25]  
            • Gradient Flow: LSTM's architecture, with its gates and cell state, helps mitigate the vanishing 
              gradient problem, which can hinder learning in simple RNNs. [22, 22, 26, 26]  

            4. Steps in Backpropagation through the Computational Graph for LSTM: [27, 28, 29]  

            • Forward Pass: Calculate the output of each gate and the cell state at each time step. 
              [27, 27, 28, 28, 30]  
            • Loss Calculation: Compute the error between the network's predictions and the actual values. [28, 28, 29, 29]  
            • Backward Pass: 
                • Start from the output layer and compute the gradient of the loss with respect to the 
                  output. [2, 5, 15]  
                • Propagate this gradient backward through the graph, applying the chain rule at each 
                  node. [5, 5, 15, 15]  
                • For each gate, calculate the gradients of the loss with respect to the gate's inputs 
                  (weights, biases, and previous hidden state). [27, 27, 28, 28]  
                • Consider the interactions between gates and the cell state during backpropagation. 
                  [11, 12]  
                • Aggregate the gradients for shared parameters across different time steps. 
                  [11, 11, 27, 27]  

            • Parameter Update: Adjust the weights and biases of the LSTM based on the calculated gradients, typically using an optimization algorithm like stochastic gradient descent. [2, 2, 31, 31]  

            5. Importance of Computational Graphs: 

            • Computational graphs provide a visual and structured way to understand and debug 
              backpropagation. [4, 9, 32]  
            • They make it easier to see how the gradients flow through the network and to identify 
              potential issues like vanishing or exploding gradients. [22, 22, 26, 26]  
            • Modern deep learning frameworks (like PyTorch and TensorFlow) use computational graphs to 
              automatically compute gradients, making it easier to train complex models. [31, 33, 34, 35, 36]  

            AI responses may include mistakes.

            [1]https://medium.com/syncedreview/bayesian-lstms-in-medicine-7af2ae13c976
            [2]https://en.wikipedia.org/wiki/Backpropagation
            [3]https://www.geeksforgeeks.org/lstm-derivation-of-back-propagation-through-time/
            [4]https://www.tutorialspoint.com/python_deep_learning/python_deep_learning_computational_graphs.htm
            [5]https://medium.com/data-science/backpropagation-in-rnn-explained-bdf853b4e1c2
            [6]https://www.ultralytics.com/glossary/backpropagation
            [7]https://medium.com/@Makleas/backpropagation-alternatives-and-neural-network-optimization-e2659ab55562
            [8]https://mehta-rohan.com/writings/blog_posts/autodiff.html
            [9]https://colah.github.io/posts/2015-08-Backprop/
            [10]https://staff.fnwi.uva.nl/r.vandenboomgaard/MachineLearning/LectureNotes/Math/automatic_differentiation.html
            [11]https://kartik2112.medium.com/lstm-back-propagation-behind-the-scenes-andrew-ng-style-notations-7207b8606cb2
            [12]https://www.researchgate.net/figure/Long-Short-Term-Memory-LSTM-has-three-layers-as-forget-gate-layer-input-gate-layer-and_fig1_337241326
            [13]https://link.springer.com/chapter/10.1007/979-8-8688-1276-7_2
            [14]https://medium.com/@eugenesh4work/litelstm-a-simpler-more-efficient-approach-to-recurrent-neural-networks-7383b66ec2a6
            [15]https://www.youtube.com/watch?v=hM74RH82LyI
            [16]https://pmc.ncbi.nlm.nih.gov/articles/PMC10803307/
            [17]https://link.springer.com/article/10.1007/s10489-021-02696-6
            [18]https://link.springer.com/content/pdf/10.1007/s00034-023-02412-4.pdf
            [19]https://ieeexplore.ieee.org/iel8/10629219/10629224/10629331.pdf
            [20]https://www.sciencedirect.com/science/article/pii/S1110016824006331
            [21]https://www.mdpi.com/2073-4441/12/2/440
            [22]https://bpb-us-e1.wpmucdn.com/sites.northeastern.edu/dist/f/94/files/2023/05/Math7243Sec12RNN.pdf
            [23]https://en.wikipedia.org/wiki/Backpropagation_through_time
            [24]https://www.sciencedirect.com/science/article/pii/S0950705124004593
            [25]https://link.springer.com/article/10.1007/s11600-025-01617-2
            [26]http://akkikiki.github.io/assets/pdf/LSTM+and+GRU.html
            [27]https://d2l.ai/chapter_recurrent-neural-networks/bptt.html
            [28]https://medium.com/@CallMeTwitch/building-a-neural-network-zoo-from-scratch-the-long-short-term-memory-network-1cec5cf31b7
            [29]https://www.youtube.com/watch?v=PmdRoZStPFM
            [30]https://journals.sagepub.com/doi/full/10.1177/14727978251321945[31] https://spotintelligence.com/2023/02/24/backpropagation/
            [32]https://medium.com/@aranya.ray1998/visualizing-backpropagation-a-journey-through-computation-graphs-4281f007f619
            [33]https://medium.com/@serverwalainfra/understanding-pytorchs-dynamic-computational-graphs-bf77ee51e5c8
            [34]https://towardsdatascience.com/computational-graphs-in-pytorch-and-tensorflow-c25cc40bdcd1/
            [35]https://medium.com/prismai/computational-graphs-in-deep-learning-d4e5f5305776
            [36]https://communities.springernature.com/posts/efficiently-solving-the-schrodinger-equation-for-many-molecules-at-once-with-deep-neural-networks
        """

        # https://www.geeksforgeeks.org/dsa/lstm-derivation-of-back-propagation-through-time/

        dz, dby = addGate.backward(self.z, by, dldy) # (w,1) * 1(w,1), (w,1) * 1(w,1) = (w,1), (w,1)
        dV, dh = mulGate.backward(V, self.h, dz) # (w,1) . (h,1).T, (w,h).T . (w,1) = (w,h), (h, 1)

        dh += dnext_h # (h,1)
        
        do = dh * tanhGate.forward(self.c) # (h,1) * np.tanh((h,1)) = (h,1)

        # dc = dh * tanhGate.backward(self.c, self.o)
        dc = self.o * tanhGate.backward(self.c, dh) # (h,1) * (1 - np.square((h,1)) * (h,1) = (h,1)
        dc += dnext_c # (h,1)

        di = dc * self.g # (h,1) * (h,1) = (h,1)
        df = dc * prev_c # (h,1) * (h,1) = (h,1)
        dg = dc * self.i # (h,1) * (h,1) = (h,1)

        # dadd2o = do * sigmoidGate.backward(self.o, np.ones_like(self.o))
        dadd2o = sigmoidGate.backward(self.o, do) # (1 - (h,1)) * (h,1) * (h,1) = (h,1)
        dadd1o, dbo = addGate.backward(self.add1o, bo, dadd2o) # (h,1) * 1(h,1), (h,1) * 1(h,1) = (h,1), (h,1)
        dmulwo, dmuluo = addGate.backward(self.mulwo, self.muluo, dadd1o) # (h,1) * 1(h,1), (h,1) * 1(h,1) = (h,1), (h,1)
        dWo, dnext_h1 = mulGate.backward(Wo, prev_h, dmulwo) # (h,1) . (h,1).T, (h,h).T . (h,1) = (h,h), (h, 1)
        dUo, _ = mulGate.backward(Uo, x, dmuluo)  # (h,1) . (w,1).T, (h,w).T . (h,1) = (h,w), (w, 1)

        # dadd2i = di * sigmoidGate.backward(self.i, np.ones_like(self.i))
        dadd2i = sigmoidGate.backward(self.i, di) # (1 - (h,1)) * (h,1) * (h,1) = (h,1)
        dadd1i, dbi = addGate.backward(self.add1i, bi, dadd2i) # (h,1) * 1(h,1), (h,1) * 1(h,1) = (h,1), (h,1)
        dmulwi, dmului = addGate.backward(self.mulwi, self.mului, dadd1i) # (h,1) * 1(h,1), (h,1) * 1(h,1) = (h,1), (h,1)
        dWi, dnext_h2 = mulGate.backward(Wi, prev_h, dmulwi) # (h,1) . (h,1).T, (h,h).T . (h,1) = (h,h), (h, 1)
        dUi, _ = mulGate.backward(Ui, x, dmului)  # (h,1) . (w,1).T, (h,w).T . (h,1) = (h,w), (w, 1)

        # dadd2f = df * sigmoidGate.backward(self.f, np.ones_like(self.f))
        dadd2f = sigmoidGate.backward(self.f, df) # (1 - (h,1)) * (h,1) * (h,1) = (h,1)
        dadd1f, dbf = addGate.backward(self.add1f, bf, dadd2f) # (h,1) * 1(h,1), (h,1) * 1(h,1) = (h,1), (h,1)
        dmulwf, dmuluf = addGate.backward(self.mulwf, self.muluf, dadd1f) # (h,1) * 1(h,1), (h,1) * 1(h,1) = (h,1), (h,1)
        dWf, dnext_h3 = mulGate.backward(Wf, prev_h, dmulwf) # (h,1) . (h,1).T, (h,h).T . (h,1) = (h,h), (h, 1)
        dUf, _ = mulGate.backward(Uf, x, dmuluf)  # (h,1) . (w,1).T, (h,w).T . (h,1) = (h,w), (w, 1)

        # dadd2g = dg * tanhGate.backward(self.g, np.ones_like(self.g))
        dadd2g = tanhGate.backward(self.g, dg)  # (h,1) * (1 - np.square((h,1)) * (h,1) = (h,1)
        dadd1g, dbg = addGate.backward(self.add1g, bg, dadd2g) # (h,1) * 1(h,1), (h,1) * 1(h,1) = (h,1), (h,1)
        dmulwg, dmulug = addGate.backward(self.mulwg, self.mulug, dadd1g) # (h,1) * 1(h,1), (h,1) * 1(h,1) = (h,1), (h,1)
        dWg, dnext_h4 = mulGate.backward(Wg, prev_h, dmulwg) # (h,1) . (h,1).T, (h,h).T . (h,1) = (h,h), (h, 1)
        dUg, _ = mulGate.backward(Ug, x, dmulug)  # (h,1) . (w,1).T, (h,w).T . (h,1) = (h,w), (w, 1)

        dnext_h = dnext_h1 + dnext_h2 + dnext_h3 + dnext_h4 # (h,1) + (h,1) + (h,1) + (h,1) =(h,1)
        dnext_c = dc * self.f # (h,1) + (h,1) = (h,1)

        return (dnext_h, dnext_c, dUi, dUf, dUo, dUg, dWi, dWf, dWo, dWg, dV, dbi, dbf, dbo, dbg, dby)
