% Embed revision control info into gateware
%
% gen_rcs_init(blk, varargin)
%
% blk = The block to be configured.
% varargin = {'varname', 'value', ...} pairs
%
% Valid varnames for this block are:
% lib_name = Name of library to extract revision info from
% lib_src = Revision source for libraries (git, svn, timestamp)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                             %
%   Center for Astronomy Signal Processing and Electronics Research           %
%   http://seti.ssl.berkeley.edu/casper/                                      %
%   Copyright (C) 2010 Andrew Martens (meerKAT)                               %
%                                                                             %
%   This program is free software; you can redistribute it and/or modify      %
%   it under the terms of the GNU General Public License as published by      %
%   the Free Software Foundation; either version 2 of the License, or         %
%   (at your option) any later version.                                       %
%                                                                             %
%   This program is distributed in the hope that it will be useful,           %
%   but WITHOUT ANY WARRANTY; without even the implied warranty of            %
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             %
%   GNU General Public License for more details.                              %
%                                                                             %
%   You should have received a copy of the GNU General Public License along   %
%   with this program; if not, write to the Free Software Foundation, Inc.,   %
%   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.               %
%                                                                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function gen_rcs_init(blk, varargin),

  clog('entering gen_rcs_init', 'trace');

  check_mask_type(blk, 'gen_rcs');
  munge_block(blk, varargin);


  defaults = {};
  lib_name = get_var('lib_name', 'defaults', defaults, varargin{:});
  lib_src = get_var('lib_src', 'defaults', defaults, varargin{:});

  system_os = determine_os();

  
  plist = textscan(path(),'%s','delimiter',pathsep);
  plist = plist{1};
  
  minlen = 100000000000;
  lib_dir = '';
  for k = 1:length(plist),
      p = plist{k};
      matches = regexp(p,lib_name,'match');
      if ~isempty(matches) & (length(p) < minlen),
          lib_dir = p;
          minlen = length(p);
      end
  end
  
  if length(lib_dir),
      disp('Found library directory at')
      lib_dir
  else,
      disp('Could not find library in path, using timestamp for revision');
      lib_name
      lib_src = 'timestamp'
  end

  lib_revision = 0;
  lib_dirty = 1;
  if strcmp(lib_src, 'svn') == 1 || strcmp(lib_src, 'git') == 1,
    [lib_result, lib_revision, lib_dirty] = get_revision_info(system_os, lib_src, lib_dir, '');
    if lib_result ~= 0,
      disp('rcs_init: Failed to get revision info for libraries from specified source, will use current time as timestamp');
      lib_src = 'timestamp';
    else

    end
  end

  lib_timestamp = 0;
  [lib_timestamp_result, lib_timestamp] = get_current_time();
  if lib_timestamp_result ~= 0,
    disp('rcs_init: Failed to get timestamp for libraries. No revision info possible!');
  end

  %set up application registers
  result = set_registers(blk, 'lib', lib_src, lib_timestamp, lib_revision, lib_dirty);
  if result ~= 0,
    disp('rcs_init: Failed to set library registers');
  end 
  if lib_dirty,
      dirty_str = 'dirty';
  else,
      dirty_str = 'clean';
  end
  annotation=sprintf('%s\n%s : %s',lib_dir,dec2hex(lib_revision),dirty_str);
  set_param(blk,'AttributesFormatString',annotation);

  clog('exiting rcs_init', 'trace');

end

function [result] = set_registers(blk, section, info_type, timestamp, revision, dirty)

  result = 1;

  if strcmp(info_type, 'git'), 
    clog(sprintf('%s revision :%x dirty :%d', section, revision, dirty),'rcs_init_debug');   
  elseif strcmp(info_type, 'svn'), 
    clog(sprintf('%s revision :%d dirty :%d', section, revision, dirty),'rcs_init_debug');   
  end
  clog(sprintf('%s timestamp :%s', section, num2str(timestamp,10)),'rcs_init_debug');   

  % set up mux and revision type
  if strcmp(info_type, 'timestamp'),
    const = 1;
    type = 0;
  elseif strcmp(info_type, 'git'),
    type = 0;
    const = 0;
  elseif strcmp(info_type, 'svn'),
    const = 0;
    type = 1;
  end
  set_param([blk, '/', section, '_type'], 'const', num2str(const));
  
  % timestamp
  % blank out most sig bit so fits in 31 bits, valid till 
  set_param([blk, '/', section, '_timestamp'], 'const', num2str(bitset(uint32(timestamp),32,0))); 

  % revision info
  set_param([blk, '/', section, '_rcs_type'], 'const', num2str(type));
  if strcmp(info_type, 'git'),
    rev = sprintf('hex2dec(''%x'')',revision);
  else
    rev = num2str(revision);
  end
 
  set_param([blk, '/', section, '_revision'], 'const', rev);
  set_param([blk, '/', section, '_dirty'], 'const', num2str(dirty));
  
  result = 0;
end

% returns current time in number of seconds since epoch
function [result, timestamp] = get_current_time()
  timestamp = round(etime(clock, [1970 1 1 0 0 0]));
  result = 0;
end

function [result, timestamp] = get_timestamp(system_os, path)
  result = 1;
  if strcmp(system_os,'windows') == 1,
  %TODO windows timestamp
    disp('rcs_init: can''t do timestamp in Windoze yet');
    clog('rcs_init: can''t do timestamp in Windoze yet','error');

  elseif strcmp(system_os, 'linux'),
    % take last modification time as timestamp
    % in seconds since epoch
    [s, r] = system(['ls -l --time-style=+%s ',path]);
    if s == 0,
      info = regexp(r,'\s+','split');
      timestamp = str2num(info{6});     
      
      if isnumeric(timestamp),
        result = 0;
      end
    else
      disp('rcs_init: failure using ''ls -l ',path,''' in Linux');
      clog('rcs_init: failure using ''ls -l ',path,''' in Linux','error');
    end     
  else
    disp(['rcs_init: unknown os ',system_os]);
    clog(['rcs_init: unknown os ',system_os],'error');
  end

  return;
end

function [system_os] = determine_os()
  %following os detection logic lifted from gen_xps_files, thanks Dave G
  [s,w] = system('uname -a');

  system_os = 'linux';
  if s ~= 0
    system_os = 'windows';
  elseif ~isempty(regexp(w,'Linux'))
    system_os = 'linux';
  end

  clog([system_os,' system detected'],'rcs_init_debug');
end

function [result, revision, dirty] = get_revision_info(system_os, rev_sys, dir, file),

  result = 1; %tough crowd
  revision = 0;
  dirty = 1;

  if strcmp(rev_sys, 'git') == 1 ,
    if strcmp(system_os,'windows') == 1,
      %TODO git in windows
      disp('rcs_init: can''t handle git in Windows yet');
      clog('rcs_init: can''t handle git in Windows yet','error');
    elseif strcmp(system_os,'linux') == 1,
      
      %get latest commit
      [s, r] = system(['cd ',dir,'; git log -n 1 --abbrev-commit ',file,'| grep commit']);
      if s == 0,
        % get first commit from log
        indices=regexp(r,[' ']);
        if length(indices) ~= 0,
          revision = uint32(0);
          % convert text to 28 bit value
          for n = 0:6,
            revision = bitor(revision, bitshift(uint32(hex2dec(r(indices(1)+n+1))), (6-n)*4));       
          end        
          result = 0;
        end

        % determine if dirty. If file, must not appear, if whole repository, must be clean
        if isempty(file), search = 'grep "nothing to commit (working directory clean)"';
        else, search = ['grep ',file,' | grep modified'];
        end
        dirty = 1;
        [s, r] = system(['cd ',dir,'; git status |', search]);
        clog(['cd ',dir,'; git status | ', search], 'rcs_init_debug');
        clog([num2str(s),' : ',r], 'rcs_init_debug');
        % failed to find modified string
        if ~isempty(file) && s == 1 && isempty(r), 
          dirty = 0;
        % git succeeded and found nothing to commit
        elseif isempty(file) && s == 0,
          dirty = 0;
        end        
      else
        disp(['rcs_init: failure using ''cd ',dir,'; git log -n 1 --abbrev-commit ',file,' | grep commit'' in Linux']);
        clog(['rcs_init: failure using ''cd ',dir,'; git log -n 1 --abbrev-commit ',file,' | grep commit'' in Linux'], 'error');
      end     

    else
      disp(['rcs_init: unrecognised os ',system_os]);
    end
  elseif strcmp(rev_sys, 'svn') == 1,
    if strcmp(system_os,'windows') == 1,
      % tortoiseSVN
      [usr_r,usr_rev] = system(sprintf('%s %s\%s', 'SubWCRev.exe', dir, file));

      if (usr_r == 0) %success
        if (~isempty(regexp(usr_rev,'Local modifications'))) % we have local mods
          dirty = 1;
        else
          dirty = 0;
        end

        temp=regexp(usr_rev,'Updated to revision (?<rev>\d+)','tokens');
        if (~isempty(temp))
          revision = str2num(temp{1}{1});
        else 
          temp=regexp(usr_rev,'Last committed at revision (?<rev>\d+)','tokens');
          if (~isempty(temp))
            revision = str2num(temp{1}{1});
          end
        end 
        clog(['tortioseSVN revision :',num2str(revision),' dirty: ',num2str(dirty)],'rcs_init_debug');   
        result = 0;
      else
        disp(['rcs_init: failure using ''SubWCRev.exe ',dir,'\',file,''' in Windoze']);
        clog(['rcs_init: failure using ''SubWCRev.exe ',dir,'\',file,''' in Windoze'],'rcs_init_debug');
      end
    elseif strcmp(system_os, 'linux'), % svn in linux
      
      %revision number
      [usr_r,usr_rev] = system(sprintf('%s %s%s', 'svn info', dir, file));
      if (usr_r == 0)
        temp=regexp(usr_rev,'Revision: (?<rev>\d+)','tokens');
        if (~isempty(temp))
            revision = str2num(temp{1}{1});
        end
        
        %modified status
        [usr_r,usr_rev] = system(sprintf('%s %s%s', 'svn status 2>/dev/null | grep ''^[AMD]'' | wc -l', dir, file));
        if (usr_r == 0)
            if (str2num(usr_rev) > 0)
                dirty = 1;
            else
                dirty = 0;
            end
        else
          disp(['rcs_init: failure using ''svn status ',dir,file,''' in Linux']);
          clog(['rcs_init: failure using ''svn status ',dir,file,''' in Linux'],'error');
        end
        if isnumeric(revision), 
          result = 0;
        end
      else
        disp(sprintf('rcs_init: failure using ''svn info ''%s%s'' in Linux\nError msg: ''%s''', dir, file, usr_rev));
        clog(sprintf('rcs_init: failure using ''svn info ''%s%s'' in Linux\nError msg: ''%s''', dir, file ,usr_rev), 'error');
      end
    else
      disp(['rcs_init: unrecognised os ',system_os]);
    end 

  else  
    disp(['rcs_init: unrecognised revision system ', rev_sys]);
    clog(['rcs_init: unrecognised revision system ', rev_sys], 'error');
  end %revision system

  return; 
end

