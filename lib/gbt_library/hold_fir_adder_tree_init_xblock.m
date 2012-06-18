function hold_fir_adder_tree_init_xblock(blk,varargin)

defaults = {'n_inputs', 3, 'add_latency', 1, 'hold_period', 1};

n_inputs = get_var('n_inputs', 'defaults', defaults, varargin{:});
add_latency = get_var('add_latency', 'defaults', defaults, varargin{:});
hold_period = get_var('hold_period', 'defaults', defaults, varargin{:});



stages = ceil(log2(n_inputs));



% add In/Out ports
sync = xInport('sync');
sync_out = xOutport('sync_out');

din = cell(1,n_inputs);
for i=1:n_inputs,
    din{i} = xInport(['din',num2str(i)]);
end
dout = xOutport('dout');

% Take care of sync
sync_delay = xBlock(struct('source', 'xbsIndex_r4/Delay','name','sync_delay'), ...
                    {'latency', stages*add_latency, ...
                     'reg_retiming', 'on'}, ...
                     {sync}, ...
                     {sync_out});




% Take care of adder tree

% If nothing to add, connect in to out
if stages==0
    dout.bind(din{1});
else
    % Make adder tree
    cur_n = n_inputs;
    stage = 0;
    blk_cnt = 0;
    blks = {};
    adder_outs = cell(1,stages);
    while cur_n > 1,
        n_adds = floor(cur_n / 2);
        n_dlys = mod(cur_n, 2);
        cur_n = n_adds + n_dlys;
        prev_blks = blks;
        blks = {};
        stage = stage + 1;
        adder_outs{stage} = cell(1,cur_n);
        for j=1:cur_n,
            blk_cnt = blk_cnt + 1;
            if j <= n_adds,
                addr = ['addr',num2str(blk_cnt)];
                blks{j} = addr;
                adder_outs{stage}{j}=xSignal(['adder_outs',num2str(stage),'_',num2str(j)]);
                if stage == 1
                    xBlock(struct('source', str2func('hold_fir_adder_init_xblock'), 'name', addr), ...
                            {[blk, '/', addr], ...
                            'add_latency', add_latency, ...
                             'hold_period', hold_period}, ...
                             {din{j*2-1}, din{j*2}, sync}, ...
                             {adder_outs{stage}{j}, []});
                else
                           xBlock(struct('source', str2func('hold_fir_adder_init_xblock'), 'name', addr), ...
                            {[blk, '/', addr], ...
                            'add_latency', add_latency, ...
                             'hold_period', hold_period}, ...
                             {adder_outs{stage-1}{j*2-1}, adder_outs{stage-1}{j*2}, sync}, ...
                             {adder_outs{stage}{j}, []});
                end
            else
                dly = ['dly',num2str(blk_cnt)];
                blks{j} = dly;
                adder_outs{stage}{j}=xSignal(['adder_outs',num2str(stage),'_',num2str(j)]);
                if stage == 1
                    xBlock(struct('source', 'xbsIndex_r4/Delay','name', dly), ...
                            {'latency', add_latency, ...
                                'reg_retiming', 'on'}, ...
                                {din{j*2-1}}, ...
                                {adder_outs{stage}{j}});
                else
                    xBlock(struct('source', 'xbsIndex_r4/Delay','name', dly), ...
                            {'latency', add_latency, ...
                                'reg_retiming', 'on'}, ...
                                {adder_outs{stage-1}{j*2-1}}, ...
                                {adder_outs{stage}{j}});
                end
            end
        end
    end
    dout.bind(adder_outs{stages}{1});
end

if ~isempty(blk) && ~strcmp(blk(1), '/')
    % When finished drawing blocks and lines, remove all unused blocks.
    clean_blocks(blk);

    % Set attribute format string (block annotation)
    annotation=sprintf('latency %d',stages*add_latency);
    set_param(blk,'AttributesFormatString',annotation);
end

end
