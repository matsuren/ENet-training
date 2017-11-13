----------------------------------------------------------------------
-- ADE20K data loader,
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset

torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------
-- ADE20K dataset:

-- get class info
local classes = {}
local conClasses = {}
local class_info_file = opt.datapath .. '/objectInfo150.txt'
local file = io.open(class_info_file)

if file then
  for line in file:lines() do
    local idx, ratio, train, val, name= unpack(line:split("\t")) 
    --         print(idx .. " : " .. name)
    if idx ~= "Idx" then
      classes[tonumber(idx)]=name 
      conClasses[tonumber(idx)]=name 
    end
  end
else
end
table.insert(classes, 1, "Unlabeled") -- Unlabeled + 150 classes
local nClasses = #classes


-- get filename info
local category_file = opt.datapath .. '/sceneCategories.txt'
local file = io.open(category_file)

train_filenames={}
test_filenames={}
local train_cnt = 0
local test_cnt = 0

if file then
  for line in file:lines() do
    local filename, scene = unpack(line:split(" ")) 
    --         print(filename .. " : " .. scene)
    if string.find(filename, 'train') then
      train_cnt = train_cnt+1
      train_filenames[train_cnt]=filename
    elseif string.find(filename, 'val') then
      test_cnt = test_cnt+1          
      test_filenames[test_cnt]=filename
    end
  end
else
end

local trsize = train_cnt --  train images
local tesize = test_cnt  -- validation images
print('==> number of train: ' .. trsize)
print('==> number of test: ' .. tesize)

-- From here #class will give number of classes even after shortening the list
-- nClasses should be used to get number of classes in original list

-- saving training histogram of classes
local histClasses = torch.Tensor(#classes):zero()

print('==> number of classes: ' .. #classes)
print('classes are:')
print(classes)

--------------------------------------------------------------------------------
print '==> loading ADE20K dataset'
local trainData, testData
local loadedFromCache = false
paths.mkdir(paths.concat(opt.cachepath, 'ADE20K'))
local ADE20KCachePath = paths.concat(opt.cachepath, 'ADE20K', 'data.t7')

if opt.cachepath ~= "none" and paths.filep(ADE20KCachePath) then
  local dataCache = torch.load(ADE20KCachePath)
  trainData = dataCache.trainData
  testData = dataCache.testData
  histClasses = dataCache.histClasses
  loadedFromCache = true
  dataCache = nil
  collectgarbage()
else
  -- initialize data structures:
  trainData = {
    data = torch.FloatTensor(trsize, opt.channels, opt.imHeight, opt.imWidth),
    labels = torch.FloatTensor(trsize, opt.labelHeight, opt.labelWidth),
    preverror = 1e10, -- a really huge number
    size = function() return trsize end
  }

  testData = {
    data = torch.FloatTensor(tesize, opt.channels, opt.imHeight, opt.imWidth),
    labels = torch.FloatTensor(tesize, opt.labelHeight, opt.labelWidth),
    preverror = 1e10, -- a really huge number
    size = function() return tesize end
  }


  print('==> loading training files');

  local dataPathRoot = opt.datapath .. '/images/training/'
  assert(paths.dirp(dataPathRoot), 'No image folder found at: ' .. opt.datapath)
  local labelPathRoot = opt.datapath .. '/annotations/training/'
  assert(paths.dirp(labelPathRoot), 'No label folder found at: ' .. opt.datapath)

  for c,filename in pairs(train_filenames) do 
    --   print(c, filename)
    local imgPath = path.join(dataPathRoot, filename .. '.jpg')

    --load training images:
    local dataTemp = image.load(imgPath,3, 'byte')
    trainData.data[c] = image.scale(dataTemp,opt.imWidth, opt.imHeight)


    -- Load training labels:
    imgPath = path.join(labelPathRoot, filename .. '.png')

    -- label image data are resized to be [1,nClasses] in [0 255] scale:
    local labelIn = image.load(imgPath, 1, 'byte')
    local labelFile = image.scale(labelIn, opt.labelWidth, opt.labelHeight, 'simple'):float()

    -- labelFile:apply(function(x) return classMap[x][1] end)
    labelFile = labelFile + 1 -- consider Unlabeled(1) It dpesn't match original labels

    -- Syntax: histc(data, bins, min, max)
    histClasses = histClasses + torch.histc(labelFile, #classes, 1, #classes)

    -- convert to int and write to data structure:
    trainData.labels[c] = labelFile

    if c % 500 == 0 then
      xlua.progress(c, trsize)
    end
    collectgarbage()
  end
  print('')

  print('==> loading testing files');


  dataPathRoot = opt.datapath .. '/images/validation/'
  assert(paths.dirp(dataPathRoot), 'No image folder found at: ' .. opt.datapath)
  labelPathRoot = opt.datapath .. '/annotations/validation/'
  assert(paths.dirp(labelPathRoot), 'No label folder found at: ' .. opt.datapath)

  -- load test images and labels:
  for c,filename in pairs(test_filenames) do 
    --   print(c, filename)
    local imgPath = path.join(dataPathRoot, filename .. '.jpg')

    --load test images:
    local dataTemp = image.load(imgPath,3, 'byte')
    testData.data[c] = image.scale(dataTemp,opt.imWidth, opt.imHeight)


    -- Load test labels:
    imgPath = path.join(labelPathRoot, filename .. '.png')

    -- label image data are resized to be [1,nClasses] in [0 255] scale:
    local labelIn = image.load(imgPath, 1, 'byte')
    local labelFile = image.scale(labelIn, opt.labelWidth, opt.labelHeight, 'simple'):float()

    -- labelFile:apply(function(x) return classMap[x][1] end)
    labelFile = labelFile + 1 -- consider Unlabeled(1) It dpesn't match original labels

    -- Syntax: histc(data, bins, min, max)
    histClasses = histClasses + torch.histc(labelFile, #classes, 1, #classes)

    -- convert to int and write to data structure:
    testData.labels[c] = labelFile

    if c % 500 == 0 then
      xlua.progress(c, trsize)
    end
    collectgarbage()
  end
end

if opt.cachepath ~= "none" and not loadedFromCache then
  print('==> saving data to cache: ' .. ADE20KCachePath)
  local dataCache = {
    trainData = trainData,
    testData = testData,
    histClasses = histClasses
  }
  torch.save(ADE20KCachePath, dataCache)
  dataCache = nil
  collectgarbage()
end

----------------------------------------------------------------------
print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

for i = 1, opt.channels do
  local trainMean = trainData.data[{ {},i }]:mean()
  local trainStd = trainData.data[{ {},i }]:std()

  local testMean = testData.data[{ {},i }]:mean()
  local testStd = testData.data[{ {},i }]:std()

  print('training data, channel-'.. i ..', mean: ' .. trainMean)
  print('training data, channel-'.. i ..', standard deviation: ' .. trainStd)

  print('test data, channel-'.. i ..', mean: ' .. testMean)
  print('test data, channel-'.. i ..', standard deviation: ' .. testStd)
end

----------------------------------------------------------------------

local classes_td = {[1] = 'classes,targets\n'}
for _,cat in pairs(classes) do
  table.insert(classes_td, cat .. ',1\n')
end

local file = io.open(paths.concat(opt.save, 'categories.txt'), 'w')
file:write(table.concat(classes_td))
file:close()

-- Exports
opt.dataClasses = classes
opt.dataconClasses  = conClasses
opt.datahistClasses = histClasses

return {
  trainData = trainData,
  testData = testData,
  mean = trainMean,
  std = trainStd
}
