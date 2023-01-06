% Compute transdiganostic symptom dimension scores
clear all
clc
% -------------- Input: --------------
%
nameofdataset = 'first'; % 'first' or 'second'
%
% ------------------------------------
assert(ismember(nameofdataset,{'first','second'}),'Check value of nameofdataset!');
% default constants
dimstr = {'AD','CIT','SW'};

% load questionnaire scores of project
load(sprintf('questionnaires_%s.mat',nameofdataset)); % loads ques_struct
% load eigenvalues from factor analysis done in (Rouault et al. 2018)
filename  = ls(fullfile('./data/factor_analysis_eigenvalues.csv'));
fulltable = readtable(filename(1:end-1));
eigens    = table2array(fulltable(:,2:end));

[quesstr,~] = strtok(table2array(fulltable(:,1)),'.');  % array of questionnaire names

% convert questionnaire names to fit project's data tables
%   zung -> depress
%   leb  -> social
for qstr = {'zung','leb'}
    idx = strfind(quesstr,qstr);
    idx = find(not(cellfun('isempty',idx)));
    if strcmpi(qstr,'zung')
        quesstr(idx) = {'depress'};
    elseif strcmpi(qstr,'leb')
        quesstr(idx) = {'social'};
    end
end

% 1. extract the z-score distribution of the log-transformed scores in original FA dataset

% load questionnaire values from participants in Rouault et al. 2018; N=497
load('ME_phase2_excludqnadata_all.mat'); % loads allqna

labelstrorig = {'zung','anxiety','ocir','leb','bis','schizo','alcohol','eat','apathy'};

allsc_orig = struct;
allsc_orig_z = [];

% extract data
for i = 1:numel(allqna)
    % go through each questionnaire in Rouault et al. (2018) dataset
    for lstr = labelstrorig
        field = getfield(allqna{i},lstr{:});
        if strcmpi(lstr{:},'leb')
            sc = field.raw.avg';
        else
            sc = field.raw';
        end
        % log transform
        if contains(lstr{:},{'ocir','leb','schizo','eat','alcohol'})
            sc = sc+1;
        end
        sc = log(sc);
        if i == 1
            allsc_orig = setfield(allsc_orig,lstr{:},nan(numel(sc),numel(allqna)));
        end
        allsc_orig.(lstr{:})(:,i) = sc;
    end
end

zparams	= struct;
for lstr = labelstrorig
    zparams = setfield(zparams,lstr{:},struct);
    [zs,zparams.(lstr{:}).mu,zparams.(lstr{:}).sigma] = zscore(allsc_orig.(lstr{:}),[],'all');
    allsc_orig_z = cat(1,allsc_orig_z,zs);
end

% 2. Convert raw score on RLVOLUNP dataset to match that of the Rouault et al. 2018
labelstr = {'depress','anxiety','ocir','social','bis','schizo','alcohol','eat','apathy'};

% ocir,leb,shizo,eat,alcohol have +1 in the scores
nsubj   = 200;
allsc   = nan(209,nsubj);
allsc_z = nan(209,nsubj);
skip  = false;
labelstruct = struct;
trigger = true;
for isubj = 1:nsubj
    if isempty(ques_struct{isubj})% | ismember(isubj,idx_excl)
        continue
    end
    for lstr = labelstr
        if ~isfield(ques_struct{isubj},lstr{:})
            skip = true;
        end
    end
    if skip
        skip = false;
        continue
    end
    for lstr = labelstr
        ind = zeros(209,1);
        idx = strfind(quesstr,lstr);
        idx = find(not(cellfun('isempty',idx)));
        ind(idx) = 1;
        if trigger
            labelstruct.(lstr{:}) = ind;
        end
        field = getfield(ques_struct{isubj},lstr{:});
        if strcmpi(lstr{:},'social')
            sc = field.avg';
        else
            sc = field.raw';
        end
    
        if contains(lstr{:},{'ocir','social','schizo','eat','alcohol'})
            sc = sc+1;
        end 
        sc = log(sc);
        allsc(logical(ind),isubj) = sc;

        if strcmpi(lstr{:},'depress')
            zmu  = zparams.('zung').mu;
            zsig = zparams.('zung').sigma;
        elseif strcmpi(lstr{:},'social')
            zmu  = zparams.('leb').mu;
            zsig = zparams.('leb').sigma;
        else
            zmu  = zparams.(lstr{:}).mu;
            zsig = zparams.(lstr{:}).sigma;
        end
        % z-score based on large dataset
        allsc_z(logical(ind),isubj) = (sc - zmu)/zsig;
    end
    trigger = false;
end

% 3. Score on psychiatric dimension (project dataset)
sc_dim = nan(nsubj,3);
sc_dim_orig = nan(size(allsc_orig_z,2),3);

% project participants onto basis of Rouault et al. 2018
for idim = 1:3
    sc_dim(:,idim) = sum(eigens(:,idim).*allsc_z,1); 
    sc_dim_orig(:,idim) = sum(eigens(:,idim).*allsc_orig_z,1);
end

% sc_dim : Columns 1-3 of dimension_scores_icar_excl_*.mat 
%          where * is 'first' or 'second'

% check correlation of AD and CIT of current project and of Rouault et al. 2018
fprintf('Correlation AD - CIT: \n')
idx = ~isnan(sc_dim(:,1));
x = sc_dim(idx,1);
y = sc_dim(idx,2);
[rho,p]  = corr(x,y,'type','Pearson');
fprintf(' Current:\n');
fprintf(' rho: %.3f; p=%.4f\n',rho,p);
disp(' ');
x = sc_dim_orig(:,1);
y = sc_dim_orig(:,2);
[rho,p]  = corr(x,y,'type','Pearson');
fprintf(' Rouault et al. 2018:\n')
fprintf(' rho: %.3f; p=%.4f\n',rho,p);

