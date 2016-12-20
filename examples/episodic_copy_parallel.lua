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

-- IPC parameters
cmd:option('-host', '127.0.0.1', 'host name of the server')
cmd:option('-port', 8080, 'port number of the server')
cmd:option('-nodeIdx', 1, 'which node is this? 1-indexed')
cmd:option('-numNodes', 1, 'total number of nodes')

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
opt.inLen = opt.nseq + opt.nspaces + opt.nseq

if opt.cuda then
  require 'cunn'
  require 'cutorch'
end

assert (opt.lstmUnit == 'nn.LSTM' or opt.lstmUnit == 'nn.AssociativeLSTM')
if opt.lstmUnit == 'nn.LSTM' then
  lstmUnit = nn.LSTM
  opt.lstmOutputSize = opt.hiddenSize
else
  lstmUnit = nn.AssociativeLSTM
  opt.lstmOutputSize = opt.hiddenSize * 2
end

local delimVal = 1
local emptyVal = 2

local optimState = {}
local timer = torch.Timer()

function setup_ipc(opt)
  ipc = require 'libipc'
  sys = require 'sys'
  Tree = require 'ipc.Tree'

  print ('Connecting server and clients on port '..tostring(opt.port))
  local client,server
  if opt.nodeIdx == 1 then
    server = ipc.server(opt.host, opt.port)
    server:clients(opt.numNodes - 1, function(client) end)
  else
    client = ipc.client(opt.host, opt.port)
  end
  tree = Tree(opt.nodeIdx, opt.numNodes, 2, server, client, 
    opt.host, opt.port + opt.nodeIdx)
end

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
  local N = inp_batch:size(1)
  -- local T_inp = inp_batch:size(2)
  local T_out = out_batch:size(2)
  local decoder_tgt = torch.LongTensor(T_out, N, 1)
  for i=1,T_out do
    decoder_tgt[i] = out_batch[{{}, i}]
  end
  -- return inp_batch:t(), torch.zeros(T_out,N,1), decoder_tgt
  return inp_batch:t(), decoder_tgt
end

local EncoderDecoder, parent = torch.class('nn.EncoderDecoder', 'nn.Module')

local function make_model(opt)
  local mod = nn.Sequential()
  mod.lstmLayers = {}
  for i=1,opt.numLayers do
    mod.lstmLayers[i] = lstmUnit(opt.lstmOutputSize, opt.hiddenSize)
    mod:add(mod.lstmLayers[i])
  end
  mod:add(nn.Linear(opt.lstmOutputSize, opt.vocabSize))
  local model = nn.Sequential()
  model:add(nn.LookupTable(opt.vocabSize, opt.lstmOutputSize))
  model:add(nn.Sequencer(mod))
  return model
end

local model = make_model(opt)

local crit = nn.SequencerCriterion(nn.CrossEntropyCriterion())

if opt.numNodes > 1 then
  setup_ipc(opt)
  -- Distribute weights to all workers
  tree.scatter(model.w)
end

local gradOutFull = torch.Tensor(opt.inLen, opt.batch_size, opt.vocabSize):zero()

if opt.cuda then
  model:cuda()
  crit:cuda()
  gradOutFull = gradOutFull:cuda()
end

local w, dw = model:getParameters()
model.w, model.dw = w, dw

local function evaluate(opt)
  for i=1,5 do
    local inp, tgt = reshape(sample_batch(opt))
    local all_outs    = model:forward(inp)
    local scored_outs = all_outs[{{opt.nseq + opt.nspaces + 1, opt.inLen}}]
    print (scored_outs)
    local _, preds_cuda = torch.max(scored_outs, 3)
    local preds = torch.LongTensor(opt.nseq, opt.batch_size)
    preds:copy(preds_cuda)
    local num_corr = torch.sum(torch.eq(preds, tgt))

    print ('INPUTS', inp:t())
    print ('OUTS', scored_outs, torch.squeeze(tgt):t())
    print ('Avg CE: ', crit:forward(scored_outs, tgt))
    print ('Num corr: ', num_corr)
  end
end

local function train()
  print ('Training...')
  for z=1,opt.num_epochs do
    epoch_sum_ce = 0
    epoch_num_correct = 0
    for y=1,opt.steps_per_epoch do
      -- evaluate()
      timer:reset()
      model.dw:zero()
      local total_ce = 0
      local num_corr = 0
      
      -- local enc_inp, dec_inp, dec_out = reshape(sample_batch(opt))
      local inp, tgt = reshape(sample_batch(opt))
      --print (inp, out)
      local all_outs    = model:forward(inp)
      local scored_outs = all_outs[{{opt.nseq + opt.nspaces + 1, opt.inLen}}]
      total_ce          = total_ce + crit:forward(scored_outs, tgt)
      local gradOutput  = crit:backward(scored_outs, tgt)
      gradOutFull[{{opt.nseq + opt.nspaces + 1, opt.inLen}}]:copy(gradOutput)
      print (scored_outs)
      print (gradOutFull:norm())
      model:backward(inp, gradOutFull)
      print (model.dw:norm())
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

      -- Combine gradients from all nodes
      if opt.numNodes > 1 then tree.allReduce(model.dw, function(a, b) return a:add(b) end) end

      -- Optimize
      optim.adam(feval, model.w, optimState)
      epoch_sum_ce = epoch_sum_ce + total_ce
    end
    if opt.nodeIdx == 1 then
      print ('Epoch '..tostring(z))
      print ('Avg CE:', epoch_sum_ce / opt.steps_per_epoch)
      -- evaluate(opt)
    end
  end
end

train()
