function sOut = xDelay(sIn,latency, name)


if(~isInt(latency))
    strError = strcat('error: delay latency must be an integer; latency= ', num2str(latency));
    throwError(strError);
end
if(latency < 0)
    strError = strcat('error: delay latency must be greater than 0; latency= ', num2str(latency));
    throwError(strError);
elseif(latency > 0)
    sOut = xSignal;
    
    bDelay = xBlock(struct('source', 'Delay', 'name', name), ...
        struct('latency', latency), ...
        {sIn}, ...
        {sOut});
    
else
    sOut = sIn;
end