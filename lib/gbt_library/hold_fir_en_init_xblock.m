function hold_fir_en_init_xblock(blk, varargin)

defaults = {'hold_period', 2};


hold_period = get_var('hold_period', 'defaults', defaults, varargin{:});


%% inports
sync_in = xInport('sync_in');

%% outports
en_out = xOutport('en_out');

%% diagram

if hold_period == 1
    
    en_out.bind(sync_in);
    return;
end

% block: untitled/Constant
xlsub1_Constant_out1 = xSignal('xlsub1_Constant_out1');
xlsub1_Constant = xBlock(struct('source', 'Constant', 'name', 'Constant'), ...
                         struct('const', 0), ...
                         {}, ...
                         {xlsub1_Constant_out1});

% block: untitled/Counter1
xlsub1_Counter1_out1 = xSignal('xlsub1_Counter1_out1');
xlsub1_Counter1 = xBlock(struct('source', 'Counter', 'name', 'Counter1'), ...
                         struct('cnt_type', 'Count Limited', ...
                                'cnt_to', 0, ...
                                'operation', 'Down', ...
                                'start_count', hold_period-1, ...
                                'n_bits', nextpow2(hold_period), ...
                                'rst', 'on'), ...
                         {sync_in}, ...
                         {xlsub1_Counter1_out1});

% block: untitled/Relational
xlsub1_Relational = xBlock(struct('source', 'Relational', 'name', 'Relational'), ...
                           [], ...
                           {xlsub1_Counter1_out1, xlsub1_Constant_out1}, ...
                           {en_out});


if ~isempty(blk) && ~strcmp(blk(1),'/')
    clean_blocks(blk);
    fmtstr=sprintf('hold period: %d\n',hold_period);
    set_param(blk,'AttributesFormatString',fmtstr);
end

end

