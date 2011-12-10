function uncram_draw(numSignals, bitWidth, binPt, signalType)
% 
% xBlock;
% numSignals=4;
% bitWidth=1;
% binPt=1;
% signalType='Signed';
% pathToBlock = 'PATH_PATH_PATH';

signalTypeGood = (strcmpi(signalType, 'signed') || ...
                  strcmpi(signalType, 'unsigned') || ...
                  strcmpi(signalType, 'boolean')); 
if(~signalTypeGood)
    throwError('The signal type must be either ''Signed'', ''Unsigned'' or ''Boolean''');
elseif(binPt <0)
   throwError('The Binal Point must be non-negative'); 
elseif(binPt <0)
    strError = strcat('The Binal Point must be non-negative; binPt = ', num2str(binPt)); 
   throwError(strError); 
elseif(binPt > bitWidth)
    strError = strcat('The Binal Point must be less than the bit width; binPt = ', num2str(binPt), '; bitWidth = ', num2str(bitWidth)); 
   throwError(strError); 
elseif(bitWidth < 1)
    strError = strcat('The bit width must be an integer greater than 0; bitWidth = ', num2str(bitWidth)); 
   throwError(strError); 
elseif(~isInt([numSignals, bitWidth, binPt]))
    strError = strcat('The number of signals, bit width and binal point must all be integers; numSignals = ', num2str(numSignals), '; bitWidth = ', num2str(bitWidth), '; binPt = ', num2str(binPt)); 
   throwError(strError); 
elseif(binPt == bitWidth)
    strError = strcat('The binal point equals the bit width, this block may not function as intended; binPt = ', num2str(binPt), '; bitWidth = ', num2str(bitWidth)); 
   throwWarning(strError);  
end

signalType = proper(signalType);

if(strcmpi(signalType, 'boolean'))
   binPt = 0; 
end

for(i= 1:numSignals)
    blockTemp=xOutport(strcat('out', num2str(i-1)));
    oOut{i} = blockTemp;
end

iIn = xInport('in');

for(i = 1:numSignals)
   blockName = strcat('signal', num2str(i));
   
   sSlice = xSlice(iIn, bitWidth, 'upper',  -1*(i-1)*bitWidth, blockName );
   
   if(binPt == 0)
       oOut{i}.bind(sSlice);
   else
       blockName = strcat('reinterpret', num2str(i));
       sOut{i} = xReinterpret(sSlice, 1, signalType, 1, binPt, blockName);
       oOut{i}.bind(sOut{i});
   end
end
