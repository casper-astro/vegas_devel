function sOut = xSlice(sIn, bitWidth, sliceMode, dist_from_endpoint, name)


if(strcmpi(sliceMode, 'lower'))
    sliceMode = 'Lower Bit Location + Width';
    sliceBase = 'LSB of Input';
    if(dist_from_endpoint < 0)
        strError = strcat('When sliceMode is ''lower'', dist_from_endpoint must be non-negative; dist_from_endpoint = ', dist_from_endpoint);
        throwError(strError);
    end
elseif(strcmpi(sliceMode, 'upper'))
    sliceBase = 'MSB of Input';
    sliceMode = 'Upper Bit Location + Width';
    if(dist_from_endpoint > 0)
        strError = strcat('When sliceMode is ''upper'', dist_from_endpoint must be non-positive; dist_from_endpoint = ', dist_from_endpoint);
        throwError(strError);
    end
else
    strError = strcat('sliceMode must be either ''upper'' or ''lower''; sliceMode = ', sliceMode);
    throwError(strError);
end



sOut = xSignal;

bSlice = xBlock(struct('source', 'Slice', 'name', name), struct( ...
    'nbits', bitWidth, 'boolean_output','off', 'mode', sliceMode, ...
    'base1', sliceBase, 'base0', sliceBase, 'bit1', dist_from_endpoint, ...
    'bit0', dist_from_endpoint), ...
    {sIn}, {sOut});