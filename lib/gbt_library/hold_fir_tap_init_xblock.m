function hold_fir_tap_init_xblock(blk, varargin)


defaults = {'mult_latency', 3, 'hold_period', 1, 'coefficient', 1.0, 'n_bits', 16, 'bin_pt', 14, 'ext_en', 'on'};


mult_latency = get_var('mult_latency', 'defaults', defaults, varargin{:});
hold_period = get_var('hold_period', 'defaults', defaults, varargin{:});
coefficient = get_var('coefficient', 'defaults', defaults, varargin{:});
n_bits = get_var('n_bits', 'defaults', defaults, varargin{:});
bin_pt = get_var('bin_pt', 'defaults', defaults, varargin{:});
ext_en = get_var('ext_en', 'defaults', defaults, varargin{:});


if coefficient > 2^(n_bits-bin_pt-1)
    disp('overflow occur! coefficients too big!');
    coeff_msg = ' overflow!';
else
    coeff_msg = '';
end

%% inports
xlsub2_data_in = xInport('data_in');

%% outports
xlsub2_mult_out = xOutport('mult_out');
xlsub2_en_out = xOutport('en_out');


if strcmp(ext_en, 'off')
    sync = xInport('sync');
    

    %% diagram
    if hold_period == 1      
        hold_tap_period_one_init_xblock(blk, xlsub2_data_in, sync, xlsub2_mult_out, xlsub2_en_out, coefficient, mult_latency, hold_period);
        return;
    end
    
    
    en = xSignal('en');
    hold_en = xBlock(struct('source', str2func('hold_en_init_xblock'), 'name', 'en_gen'), ...
        {[blk, '/en_gen'], ...
        'hold_period', hold_period}, ...
         {sync}, ...
         {en});
     
     
     % block: myfir_test/myfir_tap/Mult
    xlsub2_coefficient_out1 = xSignal('xlsub2_coefficient_out1');
    xlsub2_Relational_out1 = xSignal('xlsub2_Relational_out1');
    xlsub2_Mult = xBlock(struct('source', 'Mult', 'name', 'Mult'), ...
                         struct('en', 'on', ...
                                'latency', mult_latency), ...
                         {xlsub2_coefficient_out1, xlsub2_data_in, en}, ...
                         {xlsub2_mult_out});

    % block: myfir_test/myfir_tap/coefficient
    xlsub2_coefficient = xBlock(struct('source', 'Constant', 'name', 'coefficient'), ...
                                struct('const', coefficient, ...
                                        'n_bits', n_bits, ...
                                        'bin_pt', bin_pt, ...
                                       'explicit_period', 'on'), ...
                                {}, ...
                                {xlsub2_coefficient_out1});

    % extra outport assignment
    xConnector(xlsub2_en_out,en);
     
else

    %% inports
    en_in = xInport('en_in');

    %% diagram
    if hold_period == 1
        hold_tap_period_one_init_xblock(blk, xlsub2_data_in, en_in, xlsub2_mult_out, xlsub2_en_out, coefficient, mult_latency, hold_period, coeff_msg);
        return;
    end



    % block: myfir_test/myfir_tap/Mult
    xlsub2_coefficient_out1 = xSignal('xlsub2_coefficient_out1');
    xlsub2_Relational_out1 = xSignal('xlsub2_Relational_out1');
    xlsub2_Mult = xBlock(struct('source', 'Mult', 'name', 'Mult'), ...
                         struct('en', 'on', ...
                                'latency', mult_latency), ...
                         {xlsub2_coefficient_out1, xlsub2_data_in, en_in}, ...
                         {xlsub2_mult_out});

    % block: myfir_test/myfir_tap/coefficient
    xlsub2_coefficient = xBlock(struct('source', 'Constant', 'name', 'coefficient'), ...
                                struct('const', coefficient, ...
                                       'explicit_period', 'on'), ...
                                {}, ...
                                {xlsub2_coefficient_out1});

    % extra outport assignment
    xConnector(xlsub2_en_out, en_in);
end


if ~isempty(blk) && ~strcmp(blk(1),'/')
    fmtstr=sprintf('coefficient: %f%s\n hold period: %d\n mult latency: %d',coefficient,coeff_msg, hold_period, mult_latency);
    set_param(blk,'AttributesFormatString',fmtstr);
end
end



function hold_tap_period_one_init_xblock(blk, data_inport, sync_or_en_inport, mult_outport, en_outport, coefficient, mult_latency, hold_period, coeff_msg)
        xlsub2_coefficient_out1 = xSignal('xlsub2_coefficient_out1');
        xBlock(struct('source', 'Constant', 'name', 'coefficient'), ...
                                struct('const', coefficient, ...
                                       'explicit_period', 'on'), ...
                                {}, ...
                                {xlsub2_coefficient_out1});

        xBlock(struct('source', 'Mult', 'name', 'Mult'), ...
                         struct('en', 'off', ...
                                'latency', mult_latency), ...
                         {xlsub2_coefficient_out1, data_inport}, ...
                         {mult_outport});
                     
        % extra outport assignment
        xConnector(en_outport,sync_or_en_inport);

        if ~isempty(blk) && ~strcmp(blk(1),'/')
            fmtstr=sprintf('coefficient: %f%s\n hold period: %d\n mult latency: %d',coefficient,coeff_msg,  hold_period, mult_latency);
            set_param(blk,'AttributesFormatString',fmtstr);
        end
        return;
end
