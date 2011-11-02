function cram_draw(numInputs)

% xBlock;
% numInputs=5;
% pathToBlock = 'path:cram';

for(i= 1:numInputs)
    iDataIn{i} = xInport(strcat('in', num2str(i-1)));
end

oOut = xOutport('out');

if(numInputs > 1)
    sConcatIn = {xSignal, xSignal};
    for(i = 1:numInputs)
        blockName = strcat('reinterpret_', num2str(i)); 
        sConcatIn{i} = xReinterpret(iDataIn{i}, 1, 'Unsigned', 1, 0, blockName);
    end
    
    sConcatOut = xConcat(sConcatIn, 'concat');
    oOut.bind(sConcatOut);
else
    blockName = 'reinterpret';
        sReinterpretOut = xReinterpret(iDataIn{i}, 1, 'Unsigned', 1, 0, blockName);
        oOut.bind(sReinterpretOut);
end