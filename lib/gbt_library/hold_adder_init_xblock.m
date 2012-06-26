function hold_adder_init_xblock(blk, varargin)

defaults = {'add_latency', 1, 'hold_period', 1, 'ext_en', 'on'};

add_latency = get_var('add_latency', 'defaults', defaults, varargin{:});
hold_period = get_var('hold_period', 'defaults', defaults, varargin{:});
ext_en = get_var('ext_en', 'defaults', defaults, varargin{:});


%% inports
xlsub2_data_a = xInport('data_a');
xlsub2_data_b = xInport('data_b');

%% outports
xlsub2_sum_ab = xOutport('sum_ab');
xlsub2_en_out = xOutport('en_out');
    
    
if strcmp(ext_en, 'off')   
    %% inports
    xlsub2_sync_in = xInport('sync_in');
    
       
    if hold_period == 1
        hold_adder_period_one_init_xblock(blk,xlsub2_data_a, xlsub2_data_b, xlsub2_sync_in, xlsub2_sum_ab, xlsub2_en_out, add_latency, hold_period);
        return;
    end

    en = xSignal('en');
    hold_en = xBlock(struct('source', str2func('hold_en_init_xblock'), 'name', 'en_gen'), ...
        {[blk, '/en_gen'], ...
        'hold_period', hold_period}, ...
         {xlsub2_sync_in}, ...
         {en});
     
    % block: AddSub
    xlsub2_AddSub = xBlock(struct('source', 'AddSub', 'name', 'AddSub'), ...
                           struct('en', 'on', ...
                                  'latency', add_latency), ...
                           {xlsub2_data_b, xlsub2_data_a, en}, ...
                           {xlsub2_sum_ab});


    % extra outport assignment
    xConnector(xlsub2_en_out,en);
else

    %% inports
    xlsub2_en_in = xInport('en_in');

    %% diagram


    if hold_period == 1
        hold_adder_period_one_init_xblock(blk,xlsub2_data_a, xlsub2_data_b, xlsub2_en_in, xlsub2_sum_ab, xlsub2_en_out, add_latency, hold_period);
        return;
    end



    % block: AddSub
    xlsub2_AddSub = xBlock(struct('source', 'AddSub', 'name', 'AddSub'), ...
                           struct('en', 'on', ...
                                  'latency', add_latency), ...
                           {xlsub2_data_b, xlsub2_data_a, xlsub2_en_in}, ...
                           {xlsub2_sum_ab});


    % extra outport assignment
    xConnector(xlsub2_en_out,xlsub2_en_in);
end

if ~isempty(blk) && ~strcmp(blk(1),'/')
    clean_blocks(blk);
    fmtstr=sprintf('hold period: %d\n add latency: %d',hold_period, add_latency);
    set_param(blk,'AttributesFormatString',fmtstr);
end

end


function hold_adder_period_one_init_xblock(blk, data_a_inport, data_b_inport, sync_or_en_inport, sum_outport, en_outport, add_latency, hold_period)
        xBlock(struct('source', 'AddSub', 'name', 'AddSub'), ...
                           struct('en', 'off', ...
                                  'latency', add_latency), ...
                           {data_a_inport, data_b_inport}, ...
                           {sum_outport});


        % extra outport assignment
        xConnector(en_outport,sync_or_en_inport);

        if ~isempty(blk) && ~strcmp(blk(1),'/')
            clean_blocks(blk);
            fmtstr=sprintf('hold period: %d\n add latency: %d',hold_period, add_latency);
            set_param(blk,'AttributesFormatString',fmtstr);
        end
        return;
end
