require 'rnn'

local mytester = torch.Tester()
-- local rnntest = torch.TestSuite()
local precision = 1e-5

function LSTM_checkgrad()
   if not pcall(function() require 'optim' end) then return end

   local hiddenSize = 2
   local nIndex = 2
   -- local r = nn.GRU(hiddenSize, hiddenSize, 9999, 0.25)
   -- local r = nn.GRU(hiddenSize, hiddenSize)
   local r = nn.MuFuRu(hiddenSize, hiddenSize)
   local rnn = nn.Sequential()
   rnn:add(r)
   -- rnn:add(nn.Linear(hiddenSize, hiddenSize))
   rnn:add(nn.Linear(hiddenSize, nIndex))
   rnn:add(nn.LogSoftMax())
   rnn = nn.Recursor(rnn)

   local criterion = nn.ClassNLLCriterion()
   local inputs = torch.randn(4, 2)
   local targets = torch.Tensor{1, 2, 1, 2}:resize(4, 1)
   local parameters, grads = rnn:getParameters()
   
   function f(x)
      parameters:copy(x)
      
      -- Do the forward prop
      rnn:zeroGradParameters()
      local err = 0
      local outputs = {}
      for i = 1, inputs:size(1) do
         outputs[i] = rnn:forward(inputs[i])
         err = err + criterion:forward(outputs[i], targets[i])
      end
      for i = inputs:size(1), 1, -1 do
         local gradOutput = criterion:backward(outputs[i], targets[i])
         -- print ("backwards", gradOutput:size(), outputs[i]:size(), inputs[i]:size())
         rnn:backward(inputs[i], gradOutput)
      end
      rnn:forget()
      return err, grads
   end

   local err = optim.checkgrad(f, parameters:clone())
   print ("ERR", err)
   -- mytester:assert(err < 0.0001, "LSTM optim.checkgrad error")
end

-- mytester:add(rnntest)
-- mytester:run(tests)
LSTM_checkgrad()