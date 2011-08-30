function speadizer_vegas_init(blk, varargin)
%speadizer_vegas_init(blk, spead_msw, spead_lsw, num_items, num_payloads, pktzr_waste_time, data_input_width)

defaults = {};

if same_state(blk, 'defaults', defaults, varargin{:}), return, end
disp('Running mask script for block: speadizer_vegas_xxx')
check_mask_type(blk, 'speadizer_vegas');
munge_block(blk, varargin{:});

spead_msw = get_var('spead_msw', 'defaults', defaults, varargin{:});
spead_lsw = get_var('spead_lsw', 'defaults', defaults, varargin{:});
num_items = get_var('num_items', 'defaults', defaults, varargin{:});
num_payloads = get_var('num_payloads', 'defaults', defaults, varargin{:});
pktzr_waste_time = get_var('pktzr_waste_time', 'defaults', defaults, varargin{:});
data_input_width = get_var('data_input_width', 'defaults', defaults, varargin{:});

% Set Sub-block Parameters
%=========================

% Data input width parameters on shpacketizer
set_param([blk, '/Shpacketizer'], 'pin_width', num2str(data_input_width));

clean_blocks(blk);
save_state(blk, 'defaults', defaults, varargin{:});
