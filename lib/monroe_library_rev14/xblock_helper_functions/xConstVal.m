function sOut = xConstVal(value, arithType, nBits, binPt, name)


signalTypeGood = (strcmpi(arithType, 'signed') || ...
                  strcmpi(arithType, 'unsigned')); 
if(~signalTypeGood)
    throwError('The signal type must be either ''Signed'', ''Unsigned''');
elseif(binPt > nBits)
    strError = strcat('The Binal Point must be less than the bit width; binPt = ', num2str(binPt), '; nBits = ', num2str(nBits)); 
   throwError(strError); 
elseif(nBits < 1)
    strError = strcat('The bit width must be an integer greater than 0; nBits = ', num2str(nBits)); 
   throwError(strError); 
elseif(~isInt([nBits, binPt]))
    strError = strcat('The number of signals, bit width and binal point must all be integers; bitWidth = ', num2str(nBits), '; binPt = ', num2str(binPt)); 
   throwError(strError); 
elseif(binPt == nBits)
    strError = strcat('The binal point equals the bit width, this block may not function as intended; binPt = ', num2str(binPt), '; nBits = ', num2str(nBits)); 
   throwWarning(strError);  
end

arithType = proper(arithType);


sOut = xSignal;
bConst_Val = xBlock(struct('source', 'Constant', 'name', name),struct('const', value, 'n_bits', nBits, 'bin_pt', binPt, 'arith_type', arithType, 'explicit_period', 'on', 'period', 1), ...
        {},{sOut});
