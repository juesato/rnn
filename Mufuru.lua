------------------------------------------------------------------------
--[[ MuFuRu - Multi-Function Recurrent Unit ]]--
-- Author: Jonathan Uesato
-- License: LICENSE.2nd.txt

-- Ref. A.: http://arxiv.org/pdf/1606.03002v1.pdf

-- Notation in the comments is consistent with notation in Weissenborn
-- and Rocktaschel (2016)
-- x_t : input(t)
-- v_t : feature(t)
-- s_t : state(t) / output(t)
-- r_t : reset gate (t)
-- k_t : operation controller(t)
-- p_t : vector of weights, per operation
-- phat_t : softmax normalized weights, per operation
------------------------------------------------------------------------

local MuFuRu, parent = torch.class('nn.MuFuRu', 'nn.GRU')

local SqrtDiffLayer = nn.Sequential()
                        :add(nn.CSubTable())
                        :add(nn.Sqrt())
                        :add(nn.MulConstant(0.25))

-- all operations take a table {oldState, newState} and return newState
_operations = {
   --nn.CMaxTable(), -- max NB: doesn't deal with batched inputs
   nn.SelectTable(1), -- keep
   nn.SelectTable(2), -- replace
   -- nn.CMulTable(), -- mul
   --nn.CMinTable(), -- min NB: doesn't deal with batched inputs
   -- nn.CSubTable(), -- diff
   -- nn.Sequential():add(nn.SelectTable(1)):add(nn.MulConstant(0.0)), -- forget
   -- SqrtDiffLayer -- sqrt_diff NB: sometimes causes NaN
}
NUM_OPS = 2

function MuFuRu:__init(inputSize, outputSize, rho)
   self.num_ops = NUM_OPS
   self.inputSize = inputSize
   self.outputSize = outputSize

   parent.__init(self, inputSize, outputSize, rho or 9999)
end

-------------------------- factory methods -----------------------------
function MuFuRu:buildModel()
   -- input : {x_t, s_{t-1}}
   -- output : s_t

   -- resetGate takes {x_t, s_{t-1}} to r_t
   local resetGate = nn.Sequential()
      :add(nn.ParallelTable()
         :add(nn.Linear(self.inputSize, self.outputSize))
         :add(nn.Linear(self.outputSize, self.outputSize))
      )
      :add(nn.CAddTable())
      :add(nn.Sigmoid())

   -- Feature takes {x_t, s_{t-1}, r_t} to v_t
   local feature_vec = nn.Sequential()
      :add(nn.ConcatTable()
         :add(nn.SelectTable(1)) -- input
         :add(nn.Sequential() 
            :add(nn.NarrowTable(2,2))
            :add(nn.CMulTable())
         )
      ) -- {x_t, r_t dot s_{t-1}}
      :add(nn.ParallelTable()
         :add(nn.Linear(self.inputSize, self.outputSize))
         :add(nn.Linear(self.outputSize, self.outputSize))
      )
      :add(nn.CAddTable())
      :add(nn.Tanh())

   -- op_controller takes {x_t, s_{t-1}, r_t} to phat_t
   -- Note that reset is not used
   local op_controller = nn.Sequential()
      :add(nn.NarrowTable(1,2)) -- {x_t, s_{t-1}}
      :add(nn.ParallelTable()
         :add(nn.Linear(self.inputSize, self.num_ops))
         :add(nn.Linear(self.outputSize, self.num_ops))
      )
      :add(nn.CAddTable()) --p_t
      :add(nn.SoftMax()) --phat_t

   local all_ops = nn.ConcatTable()
   for i=1,self.num_ops do
      -- each feature is a Layer taking {x_t, s_{t-1}} to a new state,
      -- which will be linearly combined based on phat
      all_ops:add(nn.Sequential()
         :add(_operations[i])
         -- :add(nn.NaN(nn.PrintSize(), "opsnan"..i))
         -- :add(nn.PrintSize("after operations "..i))
      )
   end

   -- combine_ops takes {input, prevOutput, reset} to op weights
   local combine_ops = nn.Sequential()
      :add(nn.ConcatTable()
         :add(nn.SelectTable(3))
         :add(nn.Sequential()
            :add(nn.NarrowTable(1,2))
            :add(all_ops)
         )
      )
      :add(nn.MixtureTable())

   local cell = nn.Sequential()
      :add(nn.ConcatTable()
         :add(nn.SelectTable(1))
         :add(nn.SelectTable(2))
         :add(resetGate)
      ) -- {x_t, s_{t-1}, r_t}
      :add(nn.ConcatTable()
         :add(nn.SelectTable(2))
         :add(feature_vec)
         :add(op_controller)
      ) -- {s_{t-1}, v_t, phat_t}
      :add(combine_ops) -- s_t
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
