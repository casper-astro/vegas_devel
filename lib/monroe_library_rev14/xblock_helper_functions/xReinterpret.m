function sOut = xReinterpret(sIn, forceArithType, arithType, forceBinPt, binPt, name)


if(forceArithType == 1)
    forceArithType = 'on';
    
    
    signalTypeGood = (strcmpi(arithType, 'signed') || ...
        strcmpi(arithType, 'unsigned'));
    if(~signalTypeGood)
        throwError('The arith mode must be either ''Signed'', ''Unsigned''');
    end
else
    forceArithType = 'off';
    arithType = 'Signed';
end

if(forceBinPt == 1)
    forceBinPt = 'on';
    if((~isInt(binPt))|| binPt < 0)
        strError = strcat('The binal point must be a positive integer; binPt = ', num2str(binPt));
        throwError(strError);
    end
else
    forceBinPt = 'off';
end

arithType = proper(arithType);

sOut = xSignal;

blockTemp =     xBlock(struct('source', 'Reinterpret', 'name', name), ...
    struct('force_arith_type', forceArithType, 'arith_type', arithType, ...
    'force_bin_pt', forceBinPt, 'bin_pt', binPt), {sIn}, {sOut});



