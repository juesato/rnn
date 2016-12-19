local AssociativeLSTM, parent = torch.class("nn.AssociativeLSTM", "nn.LSTM")


function AssociativeLSTM:__init(inputSize, outputSize, rho)
   require 'nngraph'
   self.inputSize = inputSize
   self.hiddenSize = outputSize
   self.outputSize = outputSize * 2 -- C^d is represented as R^2d
   parent.__init(self, self.inputSize, self.outputSize, rho) 
end

function AssociativeLSTM.makeComplexBounded(real_node, imag_node)
   -- Implements the hard bounding function bound(h) from Associative LSTM paper
   -- Restricts the modulus of a complex number to be between 0 and 1
   local INF = 99999999
   local element_wise_norm = nn.Sqrt()(
      nn.CAddTable()({
         nn.Square()(real_node),
         nn.Square()(imag_node)
      })
   )
   local bounded_real = nn.CDivTable()({
      real_node,
      nn.Clamp(1, INF)(element_wise_norm)
   })
   local bounded_imag = nn.CDivTable()({
      imag_node,
      nn.Clamp(1, INF)(element_wise_norm)
   })
   return bounded_real, bounded_imag
end

function AssociativeLSTM:buildModel()
   -- input : {input, prevOutput, prevCell}
   -- output : {output, cell}
   assert(nngraph, "Missing nngraph package")
   
   local inputs = {}
   table.insert(inputs, nn.Identity()()) -- x
   table.insert(inputs, nn.Identity()()) -- prev_h[L]
   table.insert(inputs, nn.Identity()()) -- prev_c[L]
   
   local x, prev_h, prev_c = unpack(inputs)

   -- Calculate gates: input, forget, output (3*outputSize)
   -- Followed by input key, output key, update value (6*outputSize)
   self.i2g = nn.Linear(self.inputSize, 9*self.hiddenSize)(x)
   -- self.o2g = nn.LinearNoBias(self.hiddenSize*2, 9*self.hiddenSize)(prev_h)
   self.o2g = nn.JoinTable(2)({
      nn.LinearNoBias(self.hiddenSize*2, 7*self.hiddenSize)(prev_h),
      nn.MulConstant(0)(prev_h) -- W_hu = 0
   })
   
   local all_input_sums = nn.CAddTable()({self.i2g, self.o2g})
   local reshaped = nn.Reshape(9, self.hiddenSize)(all_input_sums)

   local n1, n2, n3, n4, n5, n6, n7, n8, n9 = nn.SplitTable(2)(reshaped):split(9)
   local gf_tmp = nn.Contiguous()(nn.Replicate(2, 2)(nn.Sigmoid()(n1)))
   local gf = nn.View(-1):setNumInputDims(2)(gf_tmp)
   local gi = nn.Sigmoid()(n2)
   local go_tmp = nn.Contiguous()(nn.Replicate(2, 2)(nn.Sigmoid()(n3)))
   local go = nn.View(-1):setNumInputDims(2)(go_tmp)
   -- local ri_real = nn.Tanh()(n4)
   -- local ri_imag = nn.Tanh()(n5)
   -- local ro_real = nn.Tanh()(n6)
   -- local ro_imag = nn.Tanh()(n7)
   -- local u_real  = nn.Tanh()(n8)
   -- local u_imag  = nn.Tanh()(n9)
   local ri_real, ri_imag = AssociativeLSTM.makeComplexBounded(n4, n5)
   local ro_real, ro_imag = AssociativeLSTM.makeComplexBounded(n6, n7)
   local u_real, u_imag   = AssociativeLSTM.makeComplexBounded(n8, n9)

   local assoc_arr_real = nn.CMulTable()({gi, u_real})
   local assoc_arr_imag = nn.CMulTable()({gi, u_imag})
   -- Still have to implement multiple cell copies
   local lookup_real = nn.CSubTable()({
      nn.CMulTable()({ri_real, assoc_arr_real}),
      nn.CMulTable()({ri_imag, assoc_arr_imag})
   })
   local lookup_imag = nn.CAddTable()({
      nn.CMulTable()({ri_real, assoc_arr_imag}),
      nn.CMulTable()({ri_imag, assoc_arr_real})   
   }) 
   local lookup = nn.JoinTable(2,2)({lookup_real, lookup_imag})
   local next_c = nn.CAddTable()({
      nn.CMulTable()({gf, prev_c}),
      lookup
   })

   local next_c_real = nn.Narrow(2, 1, self.hiddenSize)(next_c)
   local next_c_imag = nn.Narrow(2, self.hiddenSize+1, self.hiddenSize)(next_c)
   local retriv_real = nn.CSubTable()({
      nn.CMulTable()({ro_real, next_c_real}),
      nn.CMulTable()({ro_imag, next_c_imag})   
   })
   local retriv_imag = nn.CAddTable()({
      nn.CMulTable()({ro_real, next_c_imag}),
      nn.CMulTable()({ro_imag, next_c_real})
   })
   -- local retriv = nn.JoinTable(2)({retriv_real, retriv_imag})
   local retriv_real_bound, retriv_imag_bound = 
      AssociativeLSTM.makeComplexBounded(retriv_real, retriv_imag)

   -- We should average the retrieved values here

   local next_h = nn.CMulTable()({
      -- nn.Tanh()(retriv),
      nn.JoinTable(2)({retriv_real_bound, retriv_imag_bound}),
      go
   })
   local outputs = {next_h, next_c}
   local model = nn.gModule(inputs, outputs)
   return model
end

function AssociativeLSTM:buildGate()
   error"Not Implemented"
end

function AssociativeLSTM:buildInputGate()
   error"Not Implemented"
end

function AssociativeLSTM:buildForgetGate()
   error"Not Implemented"
end

function AssociativeLSTM:buildHidden()
   error"Not Implemented"
end

function AssociativeLSTM:buildCell()
   error"Not Implemented"
end   
   
function AssociativeLSTM:buildOutputGate()
   error"Not Implemented"
end
