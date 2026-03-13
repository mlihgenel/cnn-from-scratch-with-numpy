import numpy as np 

class Input():
    def forward_pass(self, inputs, training):
        self.output = inputs
        
class Dense():
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
     
    def get_parameters(self):
        return self.weights, self.biases
    
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases 
        
    # forward pass 
    def forward_pass(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
    # backward pass (backpropagation)
    def backward_pass(self, dvalues):
        # parametrelerin graydanları
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # regularizasyonların gradyanları 
        # Ağırlıkların L1 regularizasyonu 
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1 
            self.dweights += self.weight_regularizer_l1 * dL1
        # Ağırlıkların L2 regularizasyonu 
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        
        # Biasların L1 regularizasyonu 
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1 
        # Biasların L2 regularizasyonu 
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
               
        # girdilerin gradyanları 
        self.dinputs = np.dot(dvalues, self.weights.T)
        
class Dropout():
    def __init__(self, rate):
        self.rate = 1 - rate 
    
    def forward_pass(self, inputs, training=True):
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return 
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask
        
    def backward_pass(self, dvalues):
        self.dinputs = dvalues * self.binary_mask
        
class Conv():
    def __init__(self, input_shape, kernel_size, filters, stride=1, padding=0,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        input_depth, input_height, input_width = input_shape 
        self.filters = filters
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.input_depth = input_depth
        self.stride = stride
        self.padding = padding 
        
        self.output_height = int((input_height - kernel_size + (2 * padding)) / stride) + 1 
        self.output_width = int((input_width - kernel_size + (2 * padding)) / stride) + 1
        
        self.output_shape = (filters, self.output_height, self.output_width)
        
        self.kernels_shape = (filters, input_depth, kernel_size, kernel_size)
        self.weights = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(filters)
        
        # Regularization parameters
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
 
    def im2col_batch(self, inputs, kernel_size, stride, padding):
        batch, C, H, W = inputs.shape
        k = kernel_size

        # Padding
        if padding > 0:
            inputs_padded = np.pad(
                inputs,
                ((0,0), (0,0), (padding,padding), (padding,padding)),
                mode="constant"
            )
            """
            input'un boyutu (batch, C, H, W) olduğu için padding eklemek istediğimizde;
            batch ve channel' e ekleme yapmıyoruz sadece görüntünün x ve y eksenine (yani sağ-sol, üst-alt) padding uygularız.
            
            """
        else:
            inputs_padded = inputs

        H_p, W_p = inputs_padded.shape[2:]
        out_h = (H_p - k) // stride + 1
        out_w = (W_p - k) // stride + 1
        """
        convolition işleminden sonraki yükseklik ve genişlik 
        heigth, weight = (28,28) ise padding = 1 durumunda 
        H_p, W_p = (30,30) olur. 
        kernel_size = 3 ise, stride = 1 durumunda; 
        out_h, out_w = (28,28) olur.
        """

        cols = np.zeros((batch, C * k * k, out_h * out_w))
        """
        cols adında sıfırlar oluşan bir matris oluşturuyoruz 
        3 boyutlu matrisin ilk boyutu batchleri tutar
        C*k*k -> patchlerin flatten edilmiş halleridir. eğer tek kanallı olsaydı k=3 durumunda 9 olacaktı 
        out_h * out_w -> patch sayısını tutar 
        """

        # patchleri ayarlama kısmı 
        idx = 0
        for y in range(out_h):
            for x in range(out_w):
                patch = inputs_padded[:, :, y*stride:y*stride+k, x*stride:x*stride+k]
                cols[:, :, idx] = patch.reshape(batch, -1)
                idx += 1

        """
        görüntünün hem x ekseninde hem de y ekseninde gezinerek eğer padding işlemi varsa stride(atlama) kadar 
        ilerleyerek patchler oluşturulur
        indexe göre patchler flatten edilerek matirse yerleştirilir 
        
        cols[batch, patchteki eleman sayısı, patch sayısı] olacak şekilde olur 
        """
        return cols, out_h, out_w
    
    # im2col fonksiyonunun tersini yapar 
    def col2im(self, cols, input_shape, k, stride, padding, out_h, out_w):
        batch, C, H, W = input_shape
        H_p = H + 2*padding
        W_p = W + 2*padding
        
        inputs_padded = np.zeros((batch, C, H_p, W_p))
        
        idx = 0
        for y in range(out_h):
            for x in range(out_w):
                patch = cols[:, :, idx].reshape(batch, C, k, k)
                inputs_padded[:, :, y*stride:y*stride+k, x*stride:x*stride+k] += patch
                idx += 1
        
        if padding > 0:
            return inputs_padded[:, :, padding:-padding, padding:-padding]
        return inputs_padded

    def forward_pass(self, inputs, training=True):
        self.inputs = inputs
        batch, C, H, W = inputs.shape

        # im2col
        X_col, out_h, out_w = self.im2col_batch(
            inputs,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )
        self.X_col = X_col
        self.out_h = out_h
        self.out_w = out_w

        # Filtreleri reshape etme (filters, C*k*k)
        W_col = self.weights.reshape(self.filters, -1)

        # Matmul (en hızlı yöntem)
        # out: (batch, filters, out_h*out_w)
        out = np.einsum('fc,bco->bfo', W_col, X_col)

        # Bias ekleme 
        out += self.biases.reshape(1, self.filters, 1)

        # Reshape: (batch, filters, out_h, out_w)
        out = out.reshape(batch, self.filters, out_h, out_w)

        self.output = out
        return self.output
    
    def backward_pass(self, dvalues):
        batch = dvalues.shape[0]
        F = self.filters
        k = self.kernel_size
        
        # dOut_col shape: (batch, F, out_h*out_w)
        dOut_col = dvalues.reshape(batch, F, -1)
        
        # W_col shape: (F, C*k*k)
        W_col = self.weights.reshape(F, -1)
        
        # X_col was saved during forward
        X_col = self.X_col  # (batch, C*k*k, out_h*out_w)
        
        # dW: batched matmul then sum over batch
        dW_col = np.matmul(dOut_col, np.transpose(X_col, (0, 2, 1))).sum(axis=0)
        self.dweights = dW_col.reshape(self.weights.shape)
        
        # dbiases
        self.dbiases = np.sum(dvalues, axis=(0,2,3))

        # Regularization gradients for conv parameters
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        
        # dX
        WT_col = W_col.T  # (C*k*k, F)
        dX_col = np.matmul(WT_col[None, :, :], dOut_col)
        
        self.dinputs = self.col2im(
            dX_col,
            self.inputs.shape,
            k,
            self.stride,
            self.padding,
            self.out_h,
            self.out_w
        )
        
    def get_parameters(self):
        return self.weights, self.biases
    
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases 

class MaxPool2D:
    def __init__(self, pool_size=2, stride=None):
        if isinstance(pool_size, int):
            self.pool_h = pool_size
            self.pool_w = pool_size
        else:
            self.pool_h, self.pool_w = pool_size

        if stride is None:
            self.stride = self.pool_h
        else:
            self.stride = stride

    def forward_pass(self, inputs, training=True):
        if inputs.ndim != 4:
            raise ValueError("MaxPool2D expects 4D input (batch, channels, height, width).")

        self.inputs = inputs
        batch, channels, height, width = inputs.shape

        self.out_h = (height - self.pool_h) // self.stride + 1
        self.out_w = (width - self.pool_w) // self.stride + 1

        self.output = np.zeros((batch, channels, self.out_h, self.out_w), dtype=inputs.dtype)
        self.max_indices = np.zeros((batch, channels, self.out_h, self.out_w), dtype=np.int64)

        for y in range(self.out_h):
            for x in range(self.out_w):
                y0 = y * self.stride
                x0 = x * self.stride
                window = inputs[:, :, y0:y0 + self.pool_h, x0:x0 + self.pool_w]
                flat_window = window.reshape(batch, channels, -1)

                self.output[:, :, y, x] = np.max(flat_window, axis=-1)
                self.max_indices[:, :, y, x] = np.argmax(flat_window, axis=-1)

    def backward_pass(self, dvalues):
        batch, channels, _, _ = self.inputs.shape
        self.dinputs = np.zeros_like(self.inputs)

        batch_idx = np.arange(batch)[:, None]
        channel_idx = np.arange(channels)[None, :]

        for y in range(self.out_h):
            for x in range(self.out_w):
                y0 = y * self.stride
                x0 = x * self.stride

                max_idx = self.max_indices[:, :, y, x]
                max_y = max_idx // self.pool_w
                max_x = max_idx % self.pool_w

                np.add.at(
                    self.dinputs,
                    (batch_idx, channel_idx, y0 + max_y, x0 + max_x),
                    dvalues[:, :, y, x]
                )

class Flatten:
    def forward_pass(self, inputs, training=True):
        self.inputs = inputs
        if inputs.ndim == 1:
            self.output = inputs.reshape(1, -1)
        else:
            self.output = inputs.reshape(inputs.shape[0], -1)
    
    def backward_pass(self, dvalues):
        self.dinputs = dvalues.reshape(self.inputs.shape)


                
