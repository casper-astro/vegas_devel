function bulk_delay_draw(numInputs, delay_arr)
% xBlock;
% 
% numInputs = 4;
% delay_arr = 6;
% pathToBlock = 'path:bulk_delay_draw';

[vertSize, horizSize] = size(delay_arr);

if(numInputs <= 0)
    throwError('error: num_inputs must be at least 1');
elseif(vertSize ~= 1)
    throwError('error: delay_arr must be an integer or an 1-by-N array');
end



%if they just gave us a number, they want the delays to ba all the same.
if(horizSize == 1)
   delay_arr = delay_arr * ones(1,numInputs);
elseif(horizSize ~= numInputs)
        throwError('error: delay_arr must be either an integer, or a list numInputs elements long.');
end


for(i = 1:numInputs)
   sIn{i} = xInport(strcat('in', num2str(i)));
   sOut{i} = xOutport(strcat('out', num2str(i)));
   
   blockName =  strcat('delay_', num2str(i));
   sDelayed = xDelay(sIn{i}, delay_arr(i), blockName);
   sOut{i}.bind(sDelayed);
end