import numpy as np

class new_LSTM:

    # Defined the same activation functions instead of using the library because it was greatly slowing down the training time
    def sigmoid(self, X):
        return 1/(1+np.exp(-X))

    def tanh_activation(self, X):
        return np.tanh(X)

    def softmax(self, X):
        exp_X = np.exp(X)
        exp_X_sum = np.sum(exp_X,axis=1).reshape(-1,1)
        exp_X = exp_X/exp_X_sum
        return exp_X

    def tanh_derivative(self, X):
        return 1-(X**2)    
            
    def __init__(self, input, hidden, output, learning_rate):
        self.input = input
        self.hidden = hidden
        self.output = output
        self.learning_rate = learning_rate

    # Initialize parameters for the LSTM
    def init_params(self):
        mean = 0
        std = 0.01
        # Forget gate/weight matrix
        Wf = np.random.normal(mean,std,(self.input+self.hidden,self.hidden))
        # Input gate/weight matrix
        Wi  = np.random.normal(mean,std,(self.input+self.hidden,self.hidden))
        # Output gate/weight matrix
        Wo = np.random.normal(mean,std,(self.input+self.hidden,self.hidden))
        # Gate gate weights
        Wg   = np.random.normal(mean,std,(self.input+self.hidden,self.hidden))
        # Hidden state weights to output
        Wh = np.random.normal(mean,std,(self.hidden,self.output))
        # Store all weights
        params = dict()
        params['fg'] = Wf
        params['ig'] = Wi
        params['og'] = Wo
        params['gg'] = Wg
        params['hg'] = Wh
        return params

    # create a new LSTM cell
    def new_cell(self, batch, previous_activation, previous_cell, params):
        """
        batch: the batch of data to be processed
        previous_activation: the previous activation of the cell
        previous_cell: the previous cell state
        params: the parameters of the cell
        """
        # extract parameters into their own variables
        fg, ig, og, gg = params['fg'], params['ig'], params['og'], params['gg']
        # resize the batch to concatenate with the previous activation
        concat = np.concatenate((batch, previous_activation), axis=1)
        # compute the activations for each gate
        # forget gate
        f = np.matmul(concat, fg)
        f = self.sigmoid(f)
        # input gate
        i = np.matmul(concat, ig)
        i = self.sigmoid(i)
        # output gate
        o = np.matmul(concat, og)
        o = self.sigmoid(o)
        # gate gate
        g = np.matmul(concat, gg)
        g = self.tanh_activation(g)
        # compute the new cell state
        c_matrix = np.multiply(f,previous_cell) + np.multiply(i,g)
        # compute the new activation
        a_matrix = np.multiply(o, self.tanh_activation(c_matrix))
        # store activations to be used later
        activations = dict()
        activations['f'] = f
        activations['i'] = i
        activations['o'] = o
        activations['g'] = g

        return a_matrix, c_matrix, activations

    # create a new LSTM output cell
    def new_output_cell(self, activations, params):
        """
        activations: activations from the LSTM cell
        params: parameters for the LSTM
        """
        # extract hidden output into its own variable
        hg = params['hg']
        # compute the output
        output = np.matmul(activations, hg)
        output = self.softmax(output)
        return output

    # get embeddings for batch
    def get_embeddings(self, batch, embeddings):
        """
        batch: batch of data
        embeddings: embeddings for the data
        """
        # get the embeddings for the batch
        embed = np.matmul(batch, embeddings)
        return embed

    # forward propogation through the LSTM
    def forward(self, batch, params, embeddings):
        """
        batch: batch of data
        params: parameters for the LSTM
        embeddings: embeddings for the data
        """
        # get batch size
        batch_size = batch[0].shape[0]
        # store activations
        lstm_dict = dict()
        activation_dict = dict()
        cells_dict = dict()
        outputs_dict = dict()
        embeddings_dict = dict()
        # intitialize activation and cell matrices
        a_matrix = np.zeros([batch_size, self.hidden], dtype=np.float32)
        c_matrix = np.zeros([batch_size, self.hidden], dtype=np.float32)
        # store activations in dicts
        activation_dict['a0'] = a_matrix
        cells_dict['c0'] = c_matrix
        # iterate through the batch
        for i in range(len(batch)-1):
            batch_data = batch[i] 
            # get the embeddings for the batch
            batch_data = self.get_embeddings(batch_data, embeddings)
            embeddings_dict['embed'+str(i)] = batch_data
            # create a new cell
            a, c, activations = self.new_cell(batch_data, a_matrix, c_matrix, params)
            # create output cell
            output = self.new_output_cell(a, params)
            # store activations in dicts for time t
            activation_dict['a'+str(i+1)] = a
            cells_dict['c'+str(i+1)] = c
            lstm_dict['lstm'+str(i+1)] = activations
            outputs_dict['o'+str(i+1)] = output
            # update matrices
            a_matrix = a
            c_matrix = c
        
        return lstm_dict, activation_dict, cells_dict, embeddings_dict, outputs_dict

    # calculate error for an output cell
    def output_error(self, b_labels, outputs_dict, params):
        """
        b_labels: labels for the batch
        outputs_dict: dictionary of outputs for each time step
        params: dictionary of parameters
        """
        # store output errors at each time step
        output_errors = dict()
        activation_errors = dict()
        hg = params['hg']

        # iterate through the outputs
        for i in range(1, len(outputs_dict)+1):
            # get the labels for the batch
            labels = b_labels[i]
            # get labels for predictions
            predictions = outputs_dict['o'+str(i)]
            # calculate the errors for output and activation
            o_error = predictions - labels
            a_error = np.matmul(o_error, hg.T)
            # store errors in dicts
            output_errors['output_error'+str(i)] = o_error
            activation_errors['activation_error'+str(i)] = a_error
        
        return output_errors, activation_errors

    # calculate error for a LSTM cell
    def cell_error(self, output_error, next_activation_error, next_cell_error, params, activations, cell_activation, prev_activation):
        """
        output_errors: dictionary of output activation errors
        next_activation_error: the error for the next activation
        next_cell_error: the error for the next cell
        params: dictionary of parameters
        activations: dictionary of activations
        cell_activation: the activation of the cell
        prev_activation: the previous activation
        """
        # activation error from output and next activation error
        a_error = output_error + next_activation_error
        # extract activations into their own variables
        f, i, o, g = activations['f'], activations['i'], activations['o'], activations['g']
        # calculate the error for the output gate
        o_error = np.multiply(a_error, self.tanh_activation(cell_activation)) 
        o_error = np.multiply(np.multiply(o, o_error), 1-o)
        # calculate the error for the cell
        c_error = np.multiply(a_error, o)
        c_error = np.multiply(a_error, self.tanh_derivative(self.tanh_activation(cell_activation))) 
        c_error += next_cell_error
        # calculate the error for the input gate
        i_error = np.multiply(c_error, g)
        i_error = np.multiply(np.multiply(i_error, i), 1-i)
        # calculate the error for the gate gate
        g_error = np.multiply(c_error,i)
        g_error = np.multiply(g_error, self.tanh_derivative(g))
        # calculate the error for the forget gate
        f_error = np.multiply(c_error, prev_activation)
        f_error = np.multiply(np.multiply(f_error, f), 1-f)
        # calculate the error for the previous cell
        prev_cell_error = np.multiply(c_error, f)

        # extract parameters into their own variables
        fg, ig, og, gg = params['fg'], params['ig'], params['og'], params['gg']

        # get embedding activation error
        ea_error = np.matmul(f_error, fg.T) 
        ea_error += np.matmul(i_error, ig.T)
        ea_error += np.matmul(o_error, og.T) 
        ea_error += np.matmul(g_error, gg.T)

        # hidden activation error
        inputs_hidden = fg.shape[0]
        hidden = fg.shape[1]
        inputs = inputs_hidden - hidden

        # previous activation error
        prev_error = ea_error[:,inputs:]
        # embedding error
        embed_error = ea_error[:,:inputs]
        # store errors in dict
        error = dict()
        error['f_error'] = f_error
        error['i_error'] = i_error
        error['o_error'] = o_error
        error['g_error'] = g_error

        return prev_error, prev_cell_error, embed_error, error

    # calculate derivative for output cell
    def output_derivative(self, output_errors, activation_dict, params):
        """
        output_errors: dictionary of output errors
        activation_dict: dictionary of activations
        params: dictionary of parameters
        """
        # sum of derivatives at each time
        hg = np.zeros(params['hg'].shape)
        # get size of batch
        batch_size = activation_dict['a1'].shape[0]
        # iterate through the output errors
        for i in range(1, len(output_errors)+1):
            # get the output error
            o_error = output_errors['output_error'+str(i)]
            # get the activation matrix
            a_matrix = activation_dict['a'+str(i)]
            # calculate the derivative for the output cell
            hg_derivative = np.matmul(a_matrix.T, o_error)/batch_size
            # add the derivative to the params
            hg += hg_derivative
        
        return hg

    # calculate derivative for LSTM cell
    def cell_derivative(self, error, e_matrix, a_matrix):
        """
        error: dictionary of errors
        e_matrix: embedding matrix
        a_matrix: activation matrix
        """
        # extract errors into their own variables
        f_error, i_error, o_error, g_error = error['f_error'], error['i_error'], error['o_error'], error['g_error']
        # get input activations
        c_matrix = np.concatenate((e_matrix, a_matrix), axis=1)
        # get size of batch
        batch_size = e_matrix.shape[0]
        # calculate the derivatives for the cell
        fg_derivative = np.matmul(c_matrix.T, f_error)/batch_size
        ig_derivative = np.matmul(c_matrix.T, i_error)/batch_size
        og_derivative = np.matmul(c_matrix.T, o_error)/batch_size
        gg_derivative = np.matmul(c_matrix.T, g_error)/batch_size
        # store derivatives in dict
        derivatives = dict()
        derivatives['fg_derivative'] = fg_derivative
        derivatives['ig_derivative'] = ig_derivative
        derivatives['og_derivative'] = og_derivative
        derivatives['gg_derivative'] = gg_derivative
        return derivatives
    
    # backward propagation through the LSTM
    def backward(self, b_labels, embeddings_dict, lstm_dict, activation_dict, cells_dict, outputs_dict, params):
        """
        b_labels: dictionary of labels
        embeddings_dict: dictionary of embeddings
        lstm_dict: dictionary of lstm cells
        activation_dict: dictionary of activations
        cells_dict: dictionary of cell activations
        outputs_dict: dictionary of output activations
        params: dictionary of parameters
        """
        # get the output errors
        output_errors, activation_errors = self.output_error(b_labels, outputs_dict, params)
        # create dicts to store error
        lstm_errors = dict()
        e_errors = dict()
        # next activation error
        nae = np.zeros(activation_errors['activation_error1'].shape)
        # next cell error
        nce = np.zeros(activation_errors['activation_error1'].shape)
        # calculate the error for each LSTM cell
        for i in range(len(lstm_dict), 0, -1):
            # get the previous activation
            prev_activation = cells_dict['c'+str(i-1)]
            # get the cell activation
            cell_activation = cells_dict['c'+str(i)]
            # get the activation output error
            activation_error = activation_errors['activation_error'+str(i)]
            # get the activations
            activations = lstm_dict['lstm'+str(i)]
            # calculate the error for the LSTM cell
            prev_error, prev_cell_error, embed_error, lstm_error = self.cell_error(activation_error, nae, nce, params, activations, cell_activation, prev_activation)
            # store the error for the LSTM cell
            lstm_errors['lstm_error'+str(i)] = lstm_error
            # store the error for the embedding
            e_errors['e_error'+str(i-1)] = embed_error
            # update the next activation error and cell error
            nae = prev_error
            nce = prev_cell_error
        
        # Store derivatives in dict
        derivatives = dict()
        # get derivatives for the output cell
        derivatives['hg_derivative'] = self.output_derivative(output_errors, activation_dict, params)
        # Calculate cell derivatives
        # Store cell derivatives in dict
        cell_derivatives = dict()
        for i in range(1, len(lstm_errors)+1):
            # get the error
            lstm_error = lstm_errors['lstm_error'+str(i)]
            # get the embedding matrix
            e_matrix = embeddings_dict['embed'+str(i-1)]
            # get the activation matrix
            a_matrix = activation_dict['a'+str(i-1)]
            # calculate the derivatives for the cell
            cell_derivatives['cell_derivatives'+str(i)] = self.cell_derivative(lstm_error, e_matrix, a_matrix)

        # initialize derivatives to zero
        derivatives['fg_derivative'] = np.zeros(params['fg'].shape)
        derivatives['ig_derivative'] = np.zeros(params['ig'].shape)
        derivatives['og_derivative'] = np.zeros(params['og'].shape)
        derivatives['gg_derivative'] = np.zeros(params['gg'].shape)

        # add the derivatives for each cell
        for i in range(1, len(lstm_errors)+1):
            # add the derivatives for each cell
            derivatives['fg_derivative'] += cell_derivatives['cell_derivatives'+str(i)]['fg_derivative']
            derivatives['ig_derivative'] += cell_derivatives['cell_derivatives'+str(i)]['ig_derivative']
            derivatives['og_derivative'] += cell_derivatives['cell_derivatives'+str(i)]['og_derivative']
            derivatives['gg_derivative'] += cell_derivatives['cell_derivatives'+str(i)]['gg_derivative']

        return derivatives, e_errors

    # update the embeddings
    def update_embeddings(self, embeddings, e_errors, b_labels):
        """
        embeddings: dictionary of embeddings
        e_errors: dictionary of embedding errors
        b_labels: dictionary of labels
        """
        # store derivatives in dict
        e_derivatives = np.zeros(embeddings.shape)
        # get the batch size
        batch_size = b_labels[0].shape[0]
        # sum up the embedding derivatives
        for i in range(len(e_errors)):
            # calculate the derivative
            e_derivatives += np.matmul(b_labels[i].T, e_errors['e_error'+str(i)])/batch_size
        # update the embeddings
        embeddings -= (e_derivatives * self.learning_rate)
        return embeddings

    # used to calculate the loss and accuracy
    def calculate_loss_accuracy(self, b_labels, outputs_dict):
        """
        b_labels: dictionary of labels
        outputs_dict: dictionary of output activations
        """
        loss = 0
        accuracy  = 0        
        batch_size = b_labels[0].shape[0]
        
        # iterate through each output
        for i in range(1,len(outputs_dict)+1):
            #get true labels and predictions
            labels = b_labels[i]
            pred = outputs_dict['o'+str(i)]
            
            loss += np.sum((np.multiply(labels,np.log(pred)) + np.multiply(1-labels,np.log(1-pred))),axis=1).reshape(-1,1)
            accuracy  += np.array(np.argmax(labels,1)==np.argmax(pred,1),dtype=np.float32).reshape(-1,1)
        
        #calculate the loss and accuracy
        loss = np.sum(loss)*(-1/batch_size)
        accuracy  = np.sum(accuracy)/(batch_size)
        accuracy = accuracy/len(outputs_dict)
        
        return loss,accuracy

    # Adam optimizer used for increasing accuracy and decreasing loss
    # Equation for V part of optimzer: Vdw = beta1 x Vdw + (1-beta1) x (dw)
    def init_V(self, params):
        """
        parameters: dictionary of parameters
        """
        Vf = np.zeros(params['fg'].shape)
        Vi = np.zeros(params['ig'].shape)
        Vo = np.zeros(params['og'].shape)
        Vg = np.zeros(params['gg'].shape)
        Vh = np.zeros(params['hg'].shape)
        
        V_dict = dict()
        V_dict['vf'] = Vf
        V_dict['vi'] = Vi
        V_dict['vo'] = Vo
        V_dict['vg'] = Vg
        V_dict['vh'] = Vh
        return V_dict

    # Equation for S part of optimzer: Sdw = beta2 x Sdw + (1-beta2) x (dw)^2
    def init_S(self, params):
        """
        parameters: dictionary of parameters
        """
        Sf = np.zeros(params['fg'].shape)
        Si = np.zeros(params['ig'].shape)
        So = np.zeros(params['og'].shape)
        Sg = np.zeros(params['gg'].shape)
        Sh = np.zeros(params['hg'].shape)
        
        S_dict = dict()
        S_dict['sf'] = Sf
        S_dict['si'] = Si
        S_dict['so'] = So
        S_dict['sg'] = Sg
        S_dict['sh'] = Sh
        return S_dict

    # updates the parameters using Adam optimizer
    def update_parameters(self, params,derivatives,V,S, t, learning_rate, beta1, beta2):
        """
        parameters: dictionary of parameters
        derivatives: dictionary of derivatives
        V: dictionary of V parameters
        S: dictionary of S parameters
        t: time step
        learning_rate: learning rate
        beta1: beta1 parameter
        beta2: beta2 parameter
        """

        # Extract derivatives from dictionary
        derivative_fg = derivatives['fg_derivative']
        derivative_ig = derivatives['ig_derivative']
        derivative_og = derivatives['og_derivative']
        derivative_gg = derivatives['gg_derivative']
        derivative_hg = derivatives['hg_derivative']

        # Extract parameters from dictionary
        fg = params['fg']
        ig = params['ig']
        og = params['og']
        gg = params['gg']
        hg = params['hg']

        #Extract V parameters from dictionary
        vf = V['vf']
        vi = V['vi']
        vo = V['vo']
        vg = V['vg']
        vh = V['vh']

        #Extract S parameters from dictionary
        sf = S['sf']
        si = S['si']
        so = S['so']
        sg = S['sg']
        sh = S['sh']

        #Calculate the new V parameters
        vf = (beta1*vf + (1-beta1)*derivative_fg)
        vi= (beta1*vi + (1-beta1)*derivative_ig)
        vo = (beta1*vo + (1-beta1)*derivative_og)
        vg = (beta1*vg + (1-beta1)*derivative_gg)
        vh = (beta1*vh + (1-beta1)*derivative_hg)

        #calculate the new S parameters
        sf = (beta2*sf + (1-beta2)*(derivative_fg**2))
        si = (beta2*si + (1-beta2)*(derivative_ig**2))
        so = (beta2*so + (1-beta2)*(derivative_og**2))
        sg = (beta2*sg + (1-beta2)*(derivative_gg**2))
        sh = (beta2*sh + (1-beta2)*(derivative_hg**2))

        #Calulate the new parameters
        fg = fg - learning_rate*((vf)/(np.sqrt(sf) + 1e-6))
        ig = ig - learning_rate*((vi)/(np.sqrt(si) + 1e-6))
        og = og - learning_rate*((vo)/(np.sqrt(so) + 1e-6))
        gg = gg - learning_rate*((vg)/(np.sqrt(sg) + 1e-6))
        hg = hg - learning_rate*((vh)/(np.sqrt(sh) + 1e-6))
        
        #store the new parameters
        params['fg'] = fg
        params['ig'] = ig
        params['og'] = og
        params['gg'] = gg
        params['hg'] = hg

        #store the new V parameters
        V['vf'] = vf 
        V['vi'] = vi 
        V['vo'] = vo
        V['vg'] = vg
        V['vh'] = vh

        #store the new S parameters
        S['sf'] = sf
        S['si'] = si
        S['so'] = so
        S['sg'] = sg
        S['sh'] = sh

        return params,V,S 