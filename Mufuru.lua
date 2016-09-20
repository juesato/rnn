------------------------------------------------------------------------
--[[ MuFuRu - Multi-Function Recurrent Unit ]]--
-- Author: Jonathan Uesato
-- License: LICENSE.2nd.txt

-- Ref. A.: http://arxiv.org/pdf/1606.03002v1.pdf
------------------------------------------------------------------------

local MuFuRu, parent = torch.class('nn.MuFuRu', 'nn.GRU')

local SqrtDiffLayer = nn.Sequential()
                        :add(nn.CSubTable())
                        :add(nn.Sqrt())
                        :add(nn.MulConstant(0.25))

-- all operations take a table {oldState, newState} and return newState
_operations = {
   -- nn.CMaxTable(), -- max
   nn.SelectTable(1), -- keep
   nn.SelectTable(2), -- replace
   nn.CMulTable(), -- mul
   nn.CMinTable(), -- min
   -- nn.CSubTable(), -- diff
   nn.Sequential():add(nn.SelectTable(1)):add(nn.MulConstant(0.0)) -- forget
   -- SqrtDiffLayer -- sqrt_diff
}

function MuFuRu:__init(inputSize, outputSize, rho)
   self.num_ops = #_operations
   self.inputSize = inputSize
   self.outputSize = outputSize

   parent.__init(self, inputSize, outputSize, rho or 9999)

   -- build the model
   -- self.recurrentModule = self:buildModel()

   -- make it work with nn.Container
   -- self.modules[1] = self.recurrentModule
   -- self.sharedClones[1] = self.recurrentModule

   -- cached for output(0), cell(0) and gradCell(T)
   -- self.zeroTensor = torch.Tensor() 
end

-------------------------- factory methods -----------------------------
function MuFuRu:buildModel()
   -- input : {input, prevOutput}
   -- output : output

   local nonBatchDim = 2
   -- resetGate takes {input, prevOutput} to resetGate
   local resetGate = nn.Sequential()
      :add(nn.ParallelTable()
         :add(nn.Linear(self.inputSize, self.outputSize))
         :add(nn.Linear(self.outputSize, self.outputSize))
      )
      :add(nn.CAddTable())
      :add(nn.Sigmoid())

   -- Feature takes {input, prevOutput, reset} to feature
   local feature_vec = nn.Sequential()
      :add(nn.ConcatTable()
         :add(nn.SelectTable(1))
         :add(nn.Sequential()
            :add(nn.NarrowTable(2,2))
            :add(nn.CMulTable())
         )
      )
      -- join along non-batch dimension
      :add(nn.JoinTable(nonBatchDim)) -- [x_t, r dot s_t-1]
      :add(nn.Linear(self.inputSize + self.outputSize, self.outputSize))
      :add(nn.Tanh())

   -- op_controller takes {input, prevOutput, reset} to op_weights.
   -- Note that reset is not used
   local op_controller = nn.Sequential()
      :add(nn.NarrowTable(1,2)) -- take {input, prevOutput}
      :add(nn.JoinTable(nonBatchDim)) -- k_t
      :add(nn.Linear(self.inputSize + self.outputSize, self.num_ops)) --p_t
      :add(nn.SoftMax()) --p^_t

   -- all_ops takes {oldState, newState} to {newState1, newState2, ...newStateN}
   local all_ops = nn.ConcatTable()
   for i=1,self.num_ops do
      -- feature is a Layer taking {oldState, newState, op_weights} to newState
      -- NB: op_weights is an unused argument to feature to avoid the need for an
      -- extra Sequential + Narrow
      all_ops:add(nn.Sequential()
         :add(_operations[i])
         -- :add(nn.NaN(nn.PrintSize(), "opsnan"..i))
         -- :add(nn.PrintSize("after operations "..i))
      )
   end

   local debug = nn.Sequential()
      :add(nn.NarrowTable(1,2))
      -- :add(nn.NaN(nn.PrintSize(), "printsize3"))
      :add(all_ops)
      -- :add(nn.NaN(nn.PrintSize(), "printsize2"))

   -- combine_ops takes {input, prevOutput, reset} to op weights
   local combine_ops = nn.Sequential()
      :add(nn.ConcatTable()
         :add(nn.SelectTable(3))
         :add(debug)
      )
      -- :add(nn.PrintSize(), "Before Mixture")
      :add(nn.MixtureTable())

   local cell = nn.Sequential()
      :add(nn.ConcatTable()
         :add(nn.SelectTable(1))
         :add(nn.SelectTable(2))
         :add(resetGate)
      ) -- {input,prevOutput,reset}
      :add(nn.ConcatTable()
         :add(nn.SelectTable(2))
         :add(feature_vec)
         :add(op_controller)
      ) -- {prevOutput, v_t, op controller}
      :add(combine_ops)
   return cell
end

-- Factory methods are inherited from GRU

function MuFuRu:__tostring__()
   return torch.type(self)
   -- return string.format('%s(%d -> %d)', torch.type(self), self.inputSize, self.outputSize)
end

function MuFuRu:migrate(params)
   error"Migrate not supported for MuFuRu"
end
