function hold_fir_adder_init_xblock(blk, varargin)

defaults = {'add_latency', 1, 'hold_period', 1};


add_latency = get_var('add_latency', 'defaults', defaults, varargin{:});
hold_period = get_var('hold_period', 'defaults', defaults, varargin{:});


%% inports
xlsub2_data_a = xInport('data_a');
xlsub2_data_b = xInport('data_b');
xlsub2_en_in = xInport('sync_in');

%% outports
xlsub2_sum_ab = xOutport('sum_ab');
xlsub2_en_out = xOutport('en_out');

%% diagram


if hold_period == 1
    xBlock(struct('source', 'AddSub', 'name', 'AddSub'), ...
                       struct('en', 'off', ...
                              'latency', add_latency), ...
                       {xlsub2_data_b, xlsub2_data_a}, ...
                       {xlsub2_sum_ab});
                   
                   
    % extra outport assignment
    xConnector(xlsub2_en_out,xlsub2_en_in);

    if ~isempty(blk) && ~strcmp(blk(1),'/')
        clean_blocks(blk);
        fmtstr=sprintf('hold period: %d\n add latency: %d',hold_period, add_latency);
        set_param(blk,'AttributesFormatString',fmtstr);
    end
    return;
end



% block: myfir_adder_test/myfir_adder/AddSub
xlsub2_AddSub = xBlock(struct('source', 'AddSub', 'name', 'AddSub'), ...
                       struct('en', 'on', ...
                              'latency', add_latency), ...
                       {xlsub2_data_b, xlsub2_data_a, xlsub2_en_in}, ...
                       {xlsub2_sum_ab});


% extra outport assignment
xConnector(xlsub2_en_out,xlsub2_en_in);


if ~isempty(blk) && ~strcmp(blk(1),'/')
    clean_blocks(blk);
    fmtstr=sprintf('hold period: %d\n add latency: %d',hold_period, add_latency);
    set_param(blk,'AttributesFormatString',fmtstr);
end

end

