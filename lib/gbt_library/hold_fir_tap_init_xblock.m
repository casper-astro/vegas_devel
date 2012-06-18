function hold_fir_tap_init_xblock(blk, varargin)


defaults = {'mult_latency', 3, 'hold_period', 1, 'coefficient', 1.0};


mult_latency = get_var('mult_latency', 'defaults', defaults, varargin{:});
hold_period = get_var('hold_period', 'defaults', defaults, varargin{:});
coefficient = get_var('coefficient', 'defaults', defaults, varargin{:});

%% inports
xlsub2_data_in = xInport('data_in');
xlsub2_en_in = xInport('en_in');

%% outports
xlsub2_mult_out = xOutport('mult_out');
xlsub2_en_out = xOutport('en_out');

%% diagram


if hold_period == 1
    xlsub2_coefficient_out1 = xSignal('xlsub2_coefficient_out1');
    xBlock(struct('source', 'Constant', 'name', 'coefficient'), ...
                            struct('const', coefficient, ...
                                   'explicit_period', 'on'), ...
                            {}, ...
                            {xlsub2_coefficient_out1});
                   
    xBlock(struct('source', 'Mult', 'name', 'Mult'), ...
                     struct('en', 'off', ...
                            'latency', mult_latency), ...
                     {xlsub2_coefficient_out1, xlsub2_data_in}, ...
                     {xlsub2_mult_out});
                 
                 
    % extra outport assignment
    xConnector(xlsub2_en_out,xlsub2_en_in);

    if ~isempty(blk) && ~strcmp(blk(1),'/')
        fmtstr=sprintf('coefficient: %f\n hold period: %d\n mult latency: %d',coefficient, hold_period, mult_latency);
        set_param(blk,'AttributesFormatString',fmtstr);
    end
    return;
end





% block: myfir_test/myfir_tap/Mult
xlsub2_coefficient_out1 = xSignal('xlsub2_coefficient_out1');
xlsub2_Relational_out1 = xSignal('xlsub2_Relational_out1');
xlsub2_Mult = xBlock(struct('source', 'Mult', 'name', 'Mult'), ...
                     struct('en', 'on', ...
                            'latency', mult_latency), ...
                     {xlsub2_coefficient_out1, xlsub2_data_in, xlsub2_en_in}, ...
                     {xlsub2_mult_out});

% block: myfir_test/myfir_tap/coefficient
xlsub2_coefficient = xBlock(struct('source', 'Constant', 'name', 'coefficient'), ...
                            struct('const', coefficient, ...
                                   'explicit_period', 'on'), ...
                            {}, ...
                            {xlsub2_coefficient_out1});

% extra outport assignment
xConnector(xlsub2_en_out,xlsub2_en_in);

if ~isempty(blk) && ~strcmp(blk(1),'/')
    fmtstr=sprintf('coefficient: %f\n hold period: %d\n mult latency: %d',coefficient, hold_period, mult_latency);
    set_param(blk,'AttributesFormatString',fmtstr);
end
end

