function sOut = xBulkDelay(sIn, numInputs, delay_arr, name)

if(~isInt([numInputs, delay_arr]))
    throwError('the number of inputs and delay array must both be integers.');
end

if(numInputs<1)
    strError = strcat('the number of inputs and delay array must both be integers; numInputs = ',num2str(numInputs));
    throwError(strError);
end


if(length(delay_arr) == 1)
    delay_arr = delay_arr * ones(1,numInputs);
elseif(length(delay_arr) ~= numInputs)
    throwError('delay length must be either an integer or an array numInputs long');
end

for(i = 1:numInputs)
    sOut{i} = xSignal;
end

bBulkDelay = xBlock(struct('source',str2func('bulk_delay_draw'), ...
    'name', name),{numInputs, delay_arr});
bBulkDelay.bindPort(sIn, sOut);
