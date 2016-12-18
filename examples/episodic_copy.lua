require 'nn'
require 'rnn'
require 'optim'
require 'dpnn'
require '../AssociativeLSTM'

local cmd = torch.CmdLine()
-- Training parameters
cmd:option('-num_epochs', 10000, 'number of epochs')
cmd:option('-steps_per_epoch', 1000, 'minibatches per epoch')
cmd:option('-batch_size', 4, 'number of samples per minibatch')
cmd:option('-cuda', false, 'use CUDA?')

-- Model parameters
cmd:option('-hiddenSize', 128, 'number of hidden units in LSTM')
cmd:option('-vocabSize', 20, '')
cmd:option('-numLayers', 1, 'number of LSTM layers')
cmd:option('-lstmUnit', 'nn.LSTM', 'name of LSTM class')

-- Difficulty parameters
cmd:option('-nseq', 10, 'number of symbols in each sequence')
cmd:option('-nspaces', 100, 'number of empty values before making prediction')

cmd:text()
local opt = cmd:parse(arg)
opt.inLen = opt.nseq + opt.nspaces

if opt.cuda then
  require 'cunn'
  require 'cutorch'
end
if opt.lstmUnit == 'nn.LSTM' then
  lstmUnit = nn.LSTM
else
  lstmUnit = nn.AssociativeLSTM
end

local emptyVal = 1

local optimState = {}


function sample_io(opt)
  local inp = torch.LongTensor(opt.inLen):fill(emptyVal)
  local out = torch.LongTensor(opt.nseq)
  for i=1, opt.nseq do
    inp[i] = torch.random(emptyVal+1, opt.vocabSize)
    out[i] = inp[i]
  end
  return inp, out
end

function sample_batch(opt)
  local inp = torch.LongTensor(opt.batch_size, opt.inLen)
  local out = torch.LongTensor(opt.batch_size, opt.nseq)
  for i=1,opt.batch_size do
    inp[i], out[i] = sample_io(opt)
  end
  return inp, out
end

function reshape(inp_batch, out_batch)
  local decoder_inp = {}
  local N = inp_batch:size(1)
  -- local T_inp = inp_batch:size(2)
  local T_out = out_batch:size(2)
  local decoder_tgt = torch.Tensor(T_out, N, 1)
  for i=1,T_out do
    decoder_inp[i] = torch.Tensor(N, 1):zero()
    decoder_tgt[i] = out_batch[{{}, i}]
  end
  return inp_batch:t(), torch.zeros(T_out,N,1), decoder_tgt
end

local EncoderDecoder, parent = torch.class('nn.EncoderDecoder', 'nn.Module')

function EncoderDecoder:__init(opt)
  self.enc = nn.Sequential()
  local enc = nn.Sequential()
  self.enc.lstmLayers = {}
  for i=1,opt.numLayers do
    self.enc.lstmLayers[i] = lstmUnit(opt.hiddenSize, opt.hiddenSize)
    enc:add(self.enc.lstmLayers[i])
  end
  if opt.cuda then enc:cuda() end
  self.enc:add(nn.LookupTable(opt.vocabSize, opt.hiddenSize))
  -- self.enc:add(nn.PrintSize('BEFORE LSTM'))
  self.enc:add(nn.Sequencer(enc))
  -- self.enc:add(nn.PrintSize('AFTER LSTM'))
  self.dec = nn.Sequential()
  self.dec.lstmLayers = {}
  for i=1,opt.numLayers do
    if i== 1 then
      self.dec.lstmLayers[i] = lstmUnit(1, opt.hiddenSize)
    else
      self.dec.lstmLayers[i] = lstmUnit(opt.hiddenSize, opt.hiddenSize)
    end
  end

  local dec = nn.Sequential()
  for i=1,opt.numLayers do dec:add(self.dec.lstmLayers[i]) end
  dec:add(nn.Linear(opt.hiddenSize * 2, opt.vocabSize))
  self.dec:add(nn.Sequencer(dec))
end

function EncoderDecoder:parameters()
  local p, gp = {}, {}
  local enc_p, enc_gp = self.enc:parameters()
  local dec_p, dec_gp = self.dec:parameters()
  table.insert(p, enc_p)
  table.insert(p, dec_p)
  table.insert(gp, enc_gp)
  table.insert(gp, dec_gp)
  return p, gp
end

--[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
function forwardConnect(enc, dec, inLen)
   for i=1,#enc.lstmLayers do
      dec.lstmLayers[i].userPrevOutput = 
         nn.rnn.recursiveCopy(dec.lstmLayers[i].userPrevOutput, enc.lstmLayers[i].outputs[inLen])
      dec.lstmLayers[i].userPrevCell = nn.rnn.recursiveCopy(dec.lstmLayers[i].userPrevCell, enc.lstmLayers[i].cells[inLen])
   end
end

--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
function backwardConnect(enc, dec)
   for i=1,#enc.lstmLayers do
      enc.lstmLayers[i].userNextGradCell = nn.rnn.recursiveCopy(enc.lstmLayers[i].userNextGradCell, dec.lstmLayers[i].userGradPrevCell)
      enc.lstmLayers[i].gradPrevOutput = nn.rnn.recursiveCopy(enc.lstmLayers[i].gradPrevOutput, dec.lstmLayers[i].userGradPrevOutput)
   end
end

local model = nn.EncoderDecoder(opt)

local w, dw = model:getParameters()
model.w, model.dw = w, dw

local crit = nn.SequencerCriterion(nn.CrossEntropyCriterion())
if opt.cuda then crit:cuda() end

local function evaluate()
  for i=1,5 do
    local enc_inp, dec_inp, dec_out = reshape(sample_batch(opt))
    model:forward(enc_inp)
    forwardConnect(model.enc, model.dec, opt.inLen)
    print ('INPUTS', enc_inp)
    print ('OUTS', scores, dec_out)
    print ('Loss: ', crit:forward(scores, dec_out))
  end
end

local function train()
  print ('Training...')
  local gradEncOut = torch.Tensor(opt.inLen, opt.batch_size, 2 * opt.hiddenSize):zero()
  for z=1,opt.num_epochs do
    epoch_sum_ce = 0
    epoch_num_correct = 0
    for y=1,opt.steps_per_epoch do
      -- evaluate()
      model.dw:zero()
      local total_ce = 0
      local num_corr = 0
      
      local enc_inp, dec_inp, dec_out = reshape(sample_batch(opt))
      --print (inp, out)
      model.enc:forward(enc_inp)
      forwardConnect(model.enc, model.dec, opt.nseq + opt.nspaces)
      -- local _, preds_cuda = torch.max(probs, 2)
      -- local preds = torch.LongTensor(nseq):copy(preds_cuda)
      local scores = model.dec:forward(dec_inp)
      -- num_corr = num_corr + torch.sum(torch.eq(preds, out))
      total_ce = total_ce + crit:forward(scores, dec_out)
      local gradOutput = crit:backward(scores, dec_out)
      model.dec:backward(dec_inp, gradOutput)
      backwardConnect(model.enc, model.dec)
      model.enc:backward(enc_inp, gradEncOut)


      local feval = function(x)
        if x ~= model.w then model.w:copy(x) end

        -- Clip gradients if they get too large
        local gradnorm = torch.norm(model.dw)
        local clip_grads = 5
        if gradnorm > clip_grads then
          model.dw:mul(clip_grads/gradnorm)
        end

        return total_ce, model.dw
      end

      optim.adam(feval, model.w, optimState)
      epoch_sum_ce = epoch_sum_ce + total_ce
      -- epoch_num_correct = epoch_num_correct + num_corr
    end
    print ('Epoch '..tostring(z))
    -- print ('Num correct:', epoch_num_correct / steps_per_epoch, ' out of ', batch_size * nseq)
    print ('Avg CE:', epoch_sum_ce / steps_per_epoch)
  end
end

train()
