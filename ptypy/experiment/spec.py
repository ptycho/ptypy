# -*- coding: utf-8 -*-
"""\
Utility module to read spec files
Adapted from spec_read.m by Andreas Menzel (PSI)

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: see LICENSE for details.
"""
import dateutil
import dateutil.parser

global lastSpecInfo
lastSpecInfo = None


def verbose(n,s):
    """
    This function should be replaced by the real verbose class after import.
    It is here for convenience since this module has no other external dependencies.
    """
    print(s)

class SpecScan(object):
    pass

class SpecInfo(object):
    def __init__(self, spec_filename):
        self.spec_filename = spec_filename
        self.spec_file = open(spec_filename,'r')
        self.scans = {}
        self.parse()
        global lastSpecInfo
        lastSpecInfo = self

    def parse(self, rehash=False):
        """\
        Parse the spec dat-file and extract information.
        """
        f = self.spec_file
        f.seek(0) # parse can be run many times, especially if the file is still being written

        # Initialize
        motordefs = []
        motors = []
        scans = {}

        # A while loop offers more flexibility
        lnum = -1
        continue_reading = True
        while continue_reading:
            try:
                line = next(f); lnum += 1
            except StopIteration:
                break
            if line.startswith('#O0'):
                # Beginning of a motor name list
                mlist = line.split(' ',1)[1].strip().split()
                motordefs.append(lnum)
                while True:
                    line = next(f); lnum += 1
                    if not line.startswith('#O'): break
                    mlist.extend( line.split(' ',1)[1].strip().split() )
                # This updates the current list of motor names
                motors = mlist
            if line.startswith('#S'):
                # A scan
                _,scannr,scancmd = line.split(' ', 2)
                scannr = int(scannr)
                scancmd = scancmd.strip()
                if (not rehash) and scannr in self.scans:
                    #print('Skipping known scan #S %d' % scannr)
                    continue
                #print line.strip()

                # Create the structure
                scan = SpecScan()
                scan.command = scancmd
                scan.S = scancmd
                scan.lineno = lnum

                # temporary structure
                scanstr = {}
                lastlabel = ''

                # Keep reading file until empty line
                while line.strip():
                    # Get a new line, exit everything if we are
                    # at the end of a file
                    try:
                        line = next(f); lnum += 1
                    except StopIteration:
                        continue_reading = False
                        break
                    if line.startswith('#START_TIME'):
                        line = next(f); lnum += 1
                    if line.startswith('#'):
                        # We have the beginning of a new section
                        label = line[1]
                        lastlabel = label
                        entry = scanstr.get(label,[])
                        entry.append(line.split(' ',1)[1].strip())
                        scanstr[label] = entry
                    elif line.strip():
                        # This is data for the current section ("label")
                        # We will end up here if an error message is printed in
                        # the spec file. Trying to be smart about this here.
                        # It looks like all lines that do not start with "#" in
                        # a spec file contain only numbers. So we simply check
                        # if the current line can be converted to numbers
                        try:
                            [float(token) for token in line.strip().split()]
                            entry = scanstr.get(lastlabel,[])
                            entry.append(line.strip())
                            scanstr[lastlabel] = entry
                        except ValueError:
                            verbose(1,'Ignoring bad line (number %d) in spec file' % lnum)
                            verbose(1,line.strip())

                # Aborted scan?
                scan.aborted = False
                comments = scanstr.get('C', [])
                if any(c.lower().find('aborted')>0 for c in comments):
                    scan.aborted = True

                good_scan = True
                # Get date
                try:
                    scan.date = dateutil.parser.parse(scanstr['D'][0])
                    scan.D = scan.date
                except Exception as e:
                    good_scan = False
                    verbose(1, 'Error extracting date for scan number %d.' % scannr)
                    verbose(1, e.message)

                # Motors
                try:
                    motorlines = scanstr['P']
                    motorvalues = [float(x) for mline in motorlines for x in mline.split()]
                    scan.motors = dict(zip(motors, motorvalues))
                except Exception as e:
                    good_scan = False
                    verbose(1, 'Error extracting motor values for scan number %d.' % scannr)
                    verbose(1, e.message)

                # Counter names
                try:
                    scan.counternames = scanstr['L'][0].split()
                    scan.L = scan.counternames
                except Exception as e:
                    good_scan = False
                    verbose(1, 'Error extracting counter names for scan number %d.' % scannr)
                    verbose(1, e.message)

                # Data
                try:
                    data = list(zip(*[[float(x) for x in Lline.split()] for Lline in scanstr['L'][1:]]))
                    scan.data = dict(zip(scan.counternames, data))
                except Exception as e:
                    good_scan = False
                    verbose(1, 'Error extracting counter values for scan number %d.' % scannr)
                    verbose(1, e.message)

                scan.valid = good_scan
                scans[scannr] = scan
        self.scans = scans

r"""
        assert scannr > 0
        assert burst > 0
        assert multexp > 0
        scanpos = self.scans.get(scannr, None)
        if scanpos is None:
            print 'Scan not found. parsing the file again...'
            self.parse()
            scanpos = self.scans.get(scannr, None)
            if scanpos is None:
                raise RuntimeError('Scan %d not found.' % scannr)

        cmd, lineno, fstart, fend = scanpos

        # Create the structure
        scan = SpecScan()
        scan.command = cmd
        scan.S = cmd
        scan.lineno = lineno
        scan.filepos = (fstart, fend)

        # Read text block
        self.spec_file.seek(fstart)
        scanlines = self.spec_file.read(fend-fstart).split('/n')

        # temporary structure
        scanstr = {}
        lastlabel = ''
        for line in scanline:
            if line[0] == '#':
                label = line[1]
                lastlabel = label
                entry = scanstr.get(label,[])
                entry.append(line.split(' ',1)[1])
            else:
                entry = scanstr.get(lastlabel,[])
                entry.append(line)

        # Get date
        scan.date = dateutil.parser.parse(scanstr['D'][0])
        scan.D = scan.date

        # Get motor info





% here the default parameters
  scannr     = -1;
  output_arg = {'meta','motor','counter'};
  unhandled_par_error = 1;
  force_cell = 0;
% other parameters
  burstn0 = 1;
  mexpn0  = 1;

      case 'OutPut'
        output_arg = set_TextFlag(output_arg,value);
      case 'PilatusMask'
        pilatusMask = value;
        output_arg = set_TextFlag(output_arg,'+pilatus');
      case 'PilatusDir'
        pilatusDir0 = value;
        output_arg = set_TextFlag(output_arg,'+pilatus');
      case 'Cell'
        force_cell = value;
      case 'UnhandledParError'
        unhandled_par_error = value;
      otherwise
        vararg{end+1} = name;                                   %#ok<AGROW>
        vararg{end+1} = value;                                  %#ok<AGROW>
    end
  end

  specDatFile = find_specDatFile(specDatFile);

  % reading the spec file for the scans
  [scans,scanline] = spec_readscan(specDatFile, scannr);

  % reading the spec file for configuration information
  [motorc,motorline] = spec_readconfig(specDatFile);
  motorn = cell(size(scannr));
  for jj=1:numel(scans)
    motorn{jj} = motorc{find(motorline<scanline(1),1,'last')};
  end

  varargout{1} = cell(size(scans));
  % copying vararg in order to keep its content for each iteration
  vararg_1 = vararg;
  vararg_1{end+1} = 'UnhandledParError';
  vararg_1{end+1} = 0;
  for jj=1:numel(scans)
    scanstr = scans{jj}';
    arrout = regexp(scanstr{1},' +','split');
    scannr(jj) = str2double(arrout{2});

    % get the motors
    line = ~cellfun(@isempty,regexp(scanstr,'^#P'));
    arrout = reshape(transpose(strvcat((scanstr(line)))),1,[]);
    arrout = regexp(arrout,' +|(#P[0-9]+)','split');
    motorv = str2double(arrout(~cellfun(@isempty,arrout)));

    % get the counters
    line = ~cellfun(@isempty,regexp(scanstr,'^#L'));
    counters = regexp(scanstr{line},' +','split');
    counters = counters(2:end);

    % get the data
    line  = ~cellfun(@isempty,regexp(scanstr,'^[0-9+-]'));
    data = cellfun(@str2num,scanstr(line),'UniformOutput',false);
    data = cell2mat(data(:));

    % get Pilatus information if required
    if (exist('pilatusMask','var'))
      if (exist('pilatusDir0','var'))
        pilatusDir = find_pilatusDir(pilatusDir0,scannr(jj));
      else
        pilatusDir = '';
      end

      [pilatusDir, pilatusName, vararg] = ...
        find_files(strcat(pilatusDir,pilatusMask), vararg_1);

      pilatusInd = zeros(numel(pilatusName),2);
      for ii=1:numel(pilatusName)
        arrout = regexp(pilatusName(ii).name,'[_\.]','split');
        pilatusInd(ii,:) = str2double(arrout(end-2:end-1));
      end
    end

    % make a structure
    scan_structure = struct;
    if (any(strcmp(output_arg,'meta')))
      scan_structure.S = scanstr{1};

      % get the time stamp and convert it to a MATLAB-compatible date string
      line = ~cellfun(@isempty,regexp(scanstr,'^#D'));
      datestr = regexp(scanstr{line},' +','split');
      scan_structure.D = sprintf('%s-%s-%s %s', ...
        datestr{4}, ...
        datestr{3}, ...
        datestr{6}, ...
        datestr{5});
    end

    if (any(strcmp(output_arg,'motor')) || ...
        any(strcmp(output_arg,'motors')))
      for ii=1:numel(motorn{jj})
        scan_structure.(motorn{jj}{ii}) = motorv(ii);
      end
    end

    if (any(strcmp(output_arg,'counter')) || ...
        any(strcmp(output_arg,'counters')))
      if (numel(data)>0)
        for ii=1:numel(counters)
          scan_structure.(counters{ii}) = data(:,ii);
        end
      end
    end

    if (any(strcmp(output_arg,'pilatus')))
      if (burstn0>1)
        burstn = burstn0;
      elseif (isfield(scan_structure,'burstn'))
        burstn = max(ones(size(scan_structure.burstn)),scan_structure.burstn);
      else
        burstn = burstn0;
      end
      if (mexpn0>1)
        mexpn = mexpn0;
      elseif (isfield(scan_structure,'mexpn'))
        mexpn = max(ones(size(scan_structure.burstn)),scan_structure.mexpn);
      else
        mexpn = mexpn0;
      end

      if (numel(burstn)==1 && numel(mexpn)==1)
        % these pseudo motors shouldn't be scanned and are
        %   therefore presumably constant
        Pilatus = cell(size(data,1),1);
        scanInd = [transpose(floor((0:size(data,1)-1)/mexpn)), ...
          transpose(mod(0:size(data,1)-1,mexpn))];
        for ii=1:size(data,1)
          if (burstn<=1)
            indices = all(ones(size(pilatusName))'*scanInd(ii,:)==pilatusInd,2);
          else
            indices = all(ones(size(pilatusName))'*scanInd(ii,1)==pilatusInd(:,1),2);
          end
          Pilatus{ii} = {pilatusName(indices).name};
          if (numel(indices)<burstn)
            % accounting for missing frames
            % untested, as of Dec 16
            tmp = Pilatus{ii};
            Pilatus{ii} = cell(1,burstn);
            for kk=1:numel(tmp)
              Pilatus{ii}{pilatusInd(2,kk)} = tmp{kk};
            end
          end
        end
      end
      scan_structure.Pilatus = Pilatus;
      scan_structure.PilatusDir = pilatusDir;
    end

    varargout{1}{jj} = scan_structure;
  end
  if ((unhandled_par_error) && (~isempty(vararg)))
    vararg                                                      %#ok<NOPRT>
    error('Not all named parameters have been handled.');
  end
  if (numel(varargout{1})==1 && ~force_cell)
    varargout{1} = varargout{1}{1};
  end
  varargout{2} = vararg;
  return
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [varargout] = spec_readscan(specDatFile, scannr)
% check minimum number of input arguments
  if (nargin<2)
    spec_read_help(mfilename);
    error('usage ''spec_readscan(specDatFile, scannr)''');
  end

  cmd = sprintf('grep -n ''#S '' %s', specDatFile);
  [~,sysout] = system(cmd);
  arrout = regexp(sysout,'[:\n]','split');
  arrout = arrout(~cellfun(@isempty,arrout));
  arrout = transpose(reshape(arrout,2,[]));
  [s1,~] = size(arrout);

  if (any(scannr>0))
    tmp = regexp(arrout(:,2),' +','split');
    scanID = zeros(size(tmp));
    for ii=1:numel(tmp)
      scanID(ii) = str2double(tmp{ii}{2});
    end
  end

% creating a look up table where the scans begin and end
  lineI = zeros(size(scannr));
  scans = cell(size(scannr));
  for ii=1:numel(scannr)
    if (scannr(ii)<0)
      if (-scannr(ii)>size(arrout,1))
        error('There does not seem to be sufficiently many scans in ''%s''', ...
          specDatFile);
      end
      tmp = regexp(arrout(end+scannr(ii)+1,:),' +','split');
      lineI(ii) = str2double(tmp{1});
      if (scannr(ii) < -1)
        tmp = regexp(arrout(end+scannr(ii)+2,:),' +','split');
        lineF = str2double(tmp{1})-1;
      else
        cmd = sprintf('wc -l %s', specDatFile);
        [~,sysout] = system(cmd);
        lineF = str2double(regexp(sysout,'[0-9]+ ','match'));
      end
      scannr(ii) = str2double(tmp{2}(2));
    else
      tmp = find((scannr(ii)==scanID),1,'last');
      fprintf('reading ''%s''\n',arrout{tmp,2});

      if (isempty(tmp))
        fprintf('Do not find scan #S %d in ''%s''.\n', ...
          scannr(ii),specDatFile);
        continue
      end
      lineI(ii) = str2double(arrout(tmp,1));
      if (tmp<s1)
        lineF = str2double(arrout(tmp+1,1))-1;
      else
        cmd = sprintf('wc -l %s', specDatFile);
        [~,sysout] = system(cmd);
        lineF = str2double(regexp(sysout,'[0-9]+ ','match'));
      end
    end

    cmd = sprintf('head -n +%d %s | tail -n -%d', ...
      lineF, specDatFile, lineF+1-lineI(ii));

  % read the scan
    [~,sysout] = system(cmd);
    scans{ii} = regexp(sysout,'\n','split');
    scans{ii} = scans{ii}(~cellfun(@isempty,scans{ii}));
  end
  varargout{1} = scans;
  varargout{2} = lineI;
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [motors,linenr] = spec_readconfig(specDatFile)
  % finding the end of the last config
  cmd = sprintf('grep -nE ''^#O[0-9]+ '' %s', ...
      specDatFile);
  [~,sysout] = system(cmd);
  arrout = regexp(sysout,'[:\n]','split');
  arrout = arrout(~cellfun(@isempty,arrout));
  arrout = transpose(reshape(arrout,2,[]));
  [s1,~] = size(arrout);

  % devide this collection of configuration information into single chuncks
  linenr = '';
  motors = cell(0);
  for ii=1:s1
    if (~isempty(regexp(arrout{ii,2},'^#O0+','match')))
      if (isempty(linenr))
        linenr = str2double(arrout{ii,1});
      else
        linenr = vertcat(linenr, ...
          str2double(arrout{ii,1}));
      end
      motors{numel(linenr)} = regexp(arrout{ii,2},'(^#O[0-9]+ *)| *','split');
    else
      motors{numel(linenr)} = horzcat(motors{numel(linenr)}, ...
        regexp(arrout{ii,2},'(^#O[0-9]+ *)| *','split'));
    end
  end
  for ii=1:numel(linenr)
    motors{ii} = motors{ii}(~cellfun(@isempty,motors{ii}));
  end
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function specDatFile = find_specDatFile(specDatFile)
  while (exist(specDatFile,'file') ~= 2)
% if the variable specDatFile is not a complete path to a file
%   try to guess where a spec data file can be found, by
%   - look for directories called 'spec' or 'dat-files'
%   - look for files called '*.dat'
%   - take the newest one
    compare_str = specDatFile;
    fname = dir(specDatFile);
    if (exist(specDatFile,'dir'))
      if (specDatFile(end) ~= '/')
        specDatFile = strcat(specDatFile,'/');
      end

      for ii=1:numel(fname)
        if (regexp(fname(ii).name,'.dat$'))
          specDatFile = strcat(specDatFile,'*.dat');
          fname = [];
          break;
        end
      end
      for ii=1:numel(fname)
        if (strcmp(fname(ii).name,'dat-files'))
          specDatFile = strcat(specDatFile,fname(ii).name);
          fname = [];
          break;
        end
      end
      for ii=1:numel(fname)
        if (strcmp(fname(ii).name,'specES1'))
          specDatFile = strcat(specDatFile,fname(ii).name);
          break;
        end
        if (strcmp(fname(ii).name,'spec'))
          specDatFile = strcat(specDatFile,fname(ii).name);
          break;
        end
      end
    else
      if (numel(fname)>0)
        [~,ii] = max(cell2mat({fname.datenum}));
        specDatFile = regexprep(specDatFile,'\*\.dat$',fname(ii).name);
      else
        error('''%s'' cannot be found.', specDatFile);
        break
      end
    end
    if (strcmp(specDatFile,compare_str))
      break
    end
  end
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function pilatusDir = find_pilatusDir(pilatusDir,scannr)
  while (any(exist(pilatusDir,'dir') == [2,7]))
% if the variable pilatusDir is not a complete path to a file
%   try to guess where a spec data file can be found, by
%   - look for directories called 'spec' or 'dat-files'
%   - look for files called '*.dat'
%   - take the newest one
    compare_str = pilatusDir;
    fname = dir(pilatusDir);
    if (exist(pilatusDir,'dir'))
      if (pilatusDir(end) ~= '/')
        pilatusDir = strcat(pilatusDir,'/');
      end

      for ii=1:numel(fname)
        if (strcmp(fname(ii).name,sprintf('S%02d000-%02d999',floor(scannr/1e3),floor(scannr/1e3))) || ...
            strcmp(fname(ii).name,sprintf('S%05d',scannr)))
          pilatusDir = strcat(pilatusDir,fname(ii).name);
          fname = [];
          break;
        end
      end
      for ii=1:numel(fname)
        if (strcmp(fname(ii).name,'pilatus'))
          pilatusDir = strcat(pilatusDir,fname(ii).name);
          break;
        end
      end
    end
    if (strcmp(pilatusDir,compare_str))
      break
    end
  end
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [flags] = set_TextFlag(flags,value)
  if (~iscell(value))
    value = {value};
  end
  for ii=1:numel(value)
    value_i = value{ii};
    if (regexp(value_i,'^[a-zA-Z]+$'))
      flags = {value_i};
      if (ii+1<=numel(value))
        value{ii+1} = regexprep(value{ii+1},'^([a-zA-Z]*)','+$1');
      end
    elseif (regexp(value_i,'^\+[a-zA-Z]+$'))
      flags = unique([flags,value_i(2:end)]);
      if (ii+1<=numel(value))
        value{ii+1} = regexprep(value{ii+1},'^([a-zA-Z]*)','+$1');
      end
    elseif (regexp(value_i,'^-[a-zA-Z]+$'))
      flags = flags(~strcmp(flags,value_i(2:end)));
      if (ii+1<=numel(value))
        value{ii+1} = regexprep(value{ii+1},'^([a-zA-Z]*)','-$1');
      end
    end
  end
end
"""
