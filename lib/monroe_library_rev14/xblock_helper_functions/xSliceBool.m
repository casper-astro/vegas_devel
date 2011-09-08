function sOut = xSliceBool(sIn,sliceMode,dist_from_bottom, name)

if(strcmpi(sliceMode, 'lower'))
    sliceMode = 'Lower Bit Location + Width';
    if(dist_from_bottom < 0)
        strError = strcat('When sliceMode is ''lower'', dist_from_endpoint must be non-negative; dist_from_endpoint = ', dist_from_endpoint);
        throwError(strError);
    end
elseif(strcmpi(sliceMode, 'upper'))
    sliceMode = 'Upper Bit Location + Width';
    if(dist_from_bottom > 0)
        strError = strcat('When sliceMode is ''upper'', dist_from_endpoint must be non-positive; dist_from_endpoint = ', dist_from_endpoint);
        throwError(strError);
    end
else
    strError = strcat('sliceMode must be either ''upper'' or ''lower''; sliceMode = ', sliceMode);
    throwError(strError);
end


sOut = xSignal;

bSlice = xBlock(struct('source', 'Slice', 'name', name), struct( ...
    'nbits', 1, 'boolean_output','on', 'mode', sliceMode, ...
    'base1', 'MSB of Input', 'base0', 'MSB of Input', 'bit1', dist_from_bottom, ...
    'bit0', dist_from_bottom), ...
    {sIn}, {sOut});



