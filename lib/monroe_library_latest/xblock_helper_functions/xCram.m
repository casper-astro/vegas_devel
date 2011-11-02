function sOut = xCram(sIn,name)

% block: fft_stage_n_improved/fft_stage_6/delay_end
if(iscell(sIn) ~= 1)
    throwError('error: must be cell array');
end

[vertSize, concatSize] = size(sIn);
if(vertSize ~= 1)
    throwError('error: must be 1-by-N sized cell array');
end


if(concatSize == 1)
    sOut = sIn;
else
    sOut = xSignal;
    
    blockTemp = xBlock(struct('source', str2func('cram_draw'), 'name', name), {concatSize});
    blockTemp.bindPort(sIn,{sOut});

end