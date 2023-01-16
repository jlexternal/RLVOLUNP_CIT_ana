% Analyses
clear all
clc
% -------------- Input: --------------
%
nameofdataset = 'first'; % 'first' or 'second'
%
% ------------------------------------
assert(ismember(nameofdataset,{'first','second'}),'Check value of nameofdataset!');
% default constants
condstr     = {'Ref','V+','S+'};
condorder   = [3 1 2]; % order of condition presentation in paper
dimstr      = {'AD','CIT','SW'};
parstr      = {'learning rate','decay rate','learning noise','choice temperature'};

% add directories
addpath(genpath('./data'));
addpath(genpath('./toolbox'));

% loads ques_data
%   columns: [AD, CIT, SW, ICAR, exclusion]
%   Exclusion based on missing data or catch question fail
    load(sprintf('dimension_scores_icar_excl_%s.mat',nameofdataset)); % loads ques_data
dim_data = ques_data; % rename for clarity
clearvars ques_data
idx_ques = any(~isnan(dim_data(:,1:3)),2) & ~dim_data(:,5);

% Note: condition order here is 1:Ref, 2:V+, 3:S+
% load task data
load(sprintf('pcorrect_average_%s.mat',nameofdataset)); % loads pcor_raw  (subject,condition)
load(sprintf('pcorrect_curve_%s.mat',nameofdataset));   % loads pcor_cond (subject,trial,condition)
load(sprintf('prepeat_average_%s.mat',nameofdataset));  % loads prep_raw  (subject,condition)
load(sprintf('prepeat_curve_%s.mat',nameofdataset));    % loads prep_cond (subject,trial,condition)
load(sprintf('parameter_fits_%s.mat',nameofdataset));   % loads pars      (subject,parameter,condition)
load(sprintf('age_sex_%s.mat',nameofdataset));          % loads id_age_sex (age,sex)
load(sprintf('questionnaires_%s.mat',nameofdataset));   % loads ques_struct

% load retest data
if strcmpi(nameofdataset,'first')
    pcor_raw_rt = load(sprintf('pcorrect_average_retest_first.mat')); % loads pcor_raw  (subject,condition)
    prep_raw_rt = load(sprintf('prepeat_average_retest_first.mat'));  % loads prep_raw  (subject,condition)
    load(sprintf('parameter_fits_retest_first.mat'));                 % loads pars      (subject,parameter,condition)
    pcor_raw_rt = pcor_raw_rt.pcor_raw;
    prep_raw_rt = prep_raw_rt.prep_raw;
end

% binomial test for exclusion criterion 
ntrialspercond = 160;
% Input: You may try different values of ncorrect (threshold = 89)
ncorrect = 90; % one lower will pass binomial test below
passBinomialTest = binocdf(ncorrect,ntrialspercond,.5,'upper') >= 0.05;
if passBinomialTest
    fprintf('%d/%d correct trials in any condition PASSES the binomial test FOR random choices.\n',...
        ncorrect,ntrialspercond);
else
    fprintf('%d/%d correct trials in any condition FAILS the binomial test FOR random choices.\n',...
        ncorrect,ntrialspercond);
end
pcorrect_threshold = ncorrect/ntrialspercond;
idx_incl = ~(any(pcor_raw < pcorrect_threshold,2) | any(isnan(pcor_raw),2)) & idx_ques;
nsubj = sum(idx_incl);

% organize raw questionnaire scores
ques_scores = struct;
toggler = true;
for isubj = find(idx_incl == 1)'
    ques_names = fieldnames(ques_struct{isubj});
    
    for iques = 1:numel(ques_names)
        ques_name = ques_names{iques};
        
        if ismember(ques_name, {'bis','eat','schizo'})
            score = ques_struct{isubj}.(ques_name).score.total;
        else
            score = ques_struct{isubj}.(ques_name).score;
        end
        ques_scores.(ques_name)(isubj) = score;
    end
end
% reorder the field names for plotting
ques_order = {'depress', 'anxiety', 'ocir', 'social', 'bis', 'schizo', 'alcohol', 'eat', 'apathy', 'iq'};
ques_scores = orderfields(ques_scores, ques_order);
ques_names = fieldnames(ques_scores);

clearvars ncorrect pcorrect_threshold ntrialspercond passBinomialTest

%% Figure 1C (first dataset) | Supplementary Figure 1A (second dataset)
clc
% Figure 1C (1st panel) | S1A (1st panel): Values of accuracy in each condition 
fprintf('\nMedian values [& interquartile range] of participant accuracy:\n');
for icond = 1:3
    cond = condorder(icond);
    fprintf('%s: ',pad(condstr{cond},3));
    fprintf('%.2f [%.2f %.2f]\n',quantile(pcor_raw(idx_incl,cond),[.5 .25 .75]));
end

% Figure 1D (2nd panel) | S1A (2nd panel): Values of switch rate in each condition
fprintf('\nMedian values [& interquartile range] of participant switch rate:\n');
for icond = 1:3
    cond = condorder(icond);
    fprintf('%s: ',pad(condstr{cond},3));
    fprintf('%.2f [%.2f %.2f]\n',quantile(1-prep_raw(idx_incl,cond),[.5 .25 .75]));
end

% Statistics for Figure 1C/S1A (1st and 2nd panels)
cs = nchoosek(1:3,2);
for imeas = 1:2
    if imeas == 1
        fprintf('\nSigned-rank tests on p(correct)...\n');
    else
        fprintf('Signed-rank tests on p(switch)...\n');
    end
    for ic = 1:size(cs,1)
        icond = cs(ic,1);
        jcond = cs(ic,2);
        if jcond == 2
            jcond = icond;
            icond = 2;
        end
        switch imeas
        case 1
            x = pcor_raw(idx_incl,icond);
            y = pcor_raw(idx_incl,jcond);
        case 2
            x = 1-prep_raw(idx_incl,icond);
            y = 1-prep_raw(idx_incl,jcond);
        end
        [p,~,stats] = signrank(x,y);
        fprintf('%s vs %s: p=%.4f, z=%+.4f\n',condstr{icond},condstr{jcond},p,stats.zval);
    end
    disp(' ');
end

% Figure 1C (3rd panel) | S1A (3rd panel): Accuracy curves in each condition
fprintf('\nMean values (& SEM) of participant accuracy curves\n');
for icond = 1:3
    cond = condorder(icond);
    fprintf('%s:\n',condstr{cond})
    fprintf(' %.2f  ',mean(pcor_cond(idx_incl,:,cond))');
    disp(' ');
    fprintf('(%.2f) ',std(pcor_cond(idx_incl,:,cond))/sqrt(nsubj));
    disp(' ')
end

% Figure 1D (right) | S1A (4th) panel): Switch rate curves in each condition
fprintf('\nMean values (& SEM) of participant switch curves\n');
for icond = 1:3
    cond = condorder(icond);
    fprintf('%s:\n  ',condstr{cond})
    fprintf(' %.2f  ',mean(1-prep_cond(idx_incl,:,cond))');
    fprintf('\n  ');
    fprintf('(%.2f) ',std(1-prep_cond(idx_incl,:,cond))/sqrt(nsubj));
    disp(' ')
end

%% Figure 2B (first dataset) | Supplementary Figure 1B (second dataset)
clc
% Figure 2B
fprintf('\nMedian values [& interquartile range] of model fit parameters:\n');
for ipar = 1:4
    fprintf('%s\n',parstr{ipar});
    for icond = 1:3
        cond = condorder(icond);
        fprintf('  %s: ',pad(condstr{cond},3));
        fprintf('%.3f [%.3f %.3f]\n',quantile(pars(idx_incl,ipar,cond),[.5 .25 .75]));
    end
end

% Statistics for Figure 3A
fprintf('\nStatistics: condition-wise parameter differences (signed-rank tests)\n');
cs = nchoosek(1:3,2);
for ipar = 1:4
    fprintf('%s\n',parstr{ipar});
    parx = squeeze(pars(:,ipar,:));
    for ic = 1:size(cs,1)
        icond = cs(ic,1);
        jcond = cs(ic,2);
        if jcond == 2
            jcond = icond;
            icond = 2;
        end
        x = parx(idx_incl,icond);
        y = parx(idx_incl,jcond);
        
        [p,~,stats] = signrank(x,y);
        fprintf('  %s: p=%.4f, z=%+.4f\n',pad(sprintf('%s vs %s',condstr{icond},condstr{jcond}),9),p,stats.zval);
    end
    disp(' ');
end

clearvars cs ic icond imeas ipar jcond p parx stats x y

%% Figure 3 | Supplementary Figure 4

% - Raw scores for questionnaires are found in /data/questionnaires_*.mat
% - Demographic data is found in age_sex_*.mat
%       where * is 'first' or 'second'


% Code for computing the symptom dimension scores can be found in
% compute_dimensions.mat

%% Figure 4
% Note:
%   Initializing script with nameofdataset = 'first' will yield Figure 4A
%   Initializing script with nameofdataset = 'second' will yield Figure 4B
clc
varstr = dimstr;
varstr{end+1} = 'ICAR';
for imeas = 1:2
    if imeas == 1
        y = mean(pcor_raw(idx_incl,:),2);
        measstr = 'accuracy';
    elseif imeas == 2
        y = mean(1-prep_raw(idx_incl,:),2);
        measstr = 'switch rate';
    end
    age  = id_age_sex.age(idx_incl);
    sex  = id_age_sex.sex(idx_incl);
    ad   = tiedrank(dim_data(idx_incl,1));
    cit  = tiedrank(dim_data(idx_incl,2));
    sw   = tiedrank(dim_data(idx_incl,3));
    icar = tiedrank(dim_data(idx_incl,4));
    y    = tiedrank(y);

    X   = table(age,sex,ad,cit,sw,icar,y);
    mdl = fitglm(X);
    if strcmpi(nameofdataset,'first') 
        x   = mdl.Coefficients.Estimate(5:8);
        p   = mdl.Coefficients.pValue(5:8);
        t   = mdl.Coefficients.tStat(5:8);
        SE  = mdl.Coefficients.SE(5:8);
    elseif strcmpi(nameofdataset,'second') 
        x   = mdl.Coefficients.Estimate(4:7);
        p   = mdl.Coefficients.pValue(4:7);
        t   = mdl.Coefficients.tStat(4:7);
        SE  = mdl.Coefficients.SE(4:7);
    end
    CI  = SE * 1.959964;

    fprintf('Regression (%s):\n',measstr);
    for ivar = 1:4 % ad, cit, sw, icar
        fprintf('%s: %+.2f ± %.2f; t=%+.4f ',pad(varstr{ivar},5),x(ivar),CI(ivar),t(ivar));
        if strcmpi(nameofdataset,'first') 
            fprintf(' (p=%.4f)',p(ivar)); % p-values for first dataset
        else
            % 
            % T-values and degrees of freedom (i.e. number of participants)
            % are entered in the R code for replication Bayes Factor values 
            % on the second dataset (bayesFactorCalculations.R)
            %
        end
        disp(' ')
    end
    [r,p] = corr(y,dim_data(idx_incl,2),'Type','Spearman');
    fprintf('Correlation %s-%s: r=%+.2f, p=%.4f\n','CIT',measstr,r,p);
    disp(' ');
end

%% Figure 5
% Note:
%   Initializing script with nameofdataset = 'first' will yield Figure 5A
%   Initializing script with nameofdataset = 'second' will yield Figure 5B
clc
varstr = dimstr;
varstr{end+1} = 'ICAR';
for imeas = 1:2
    if imeas == 1
        ipar = 3;
        measstr = 'learning noise';
    elseif imeas == 2
        ipar = 4;
        measstr = 'choice temperature';
    end
    y    = mean(pars(idx_incl,ipar,:),3);

    age  = id_age_sex.age(idx_incl);
    sex  = id_age_sex.sex(idx_incl);
    ad   = tiedrank(dim_data(idx_incl,1));
    cit  = tiedrank(dim_data(idx_incl,2));
    sw   = tiedrank(dim_data(idx_incl,3));
    icar = tiedrank(dim_data(idx_incl,4));
    y    = tiedrank(y);

    % regressions
    X   = table(age,sex,ad,cit,sw,icar,y);
    mdl = fitglm(X);
    if strcmpi(nameofdataset,'first') 
        x   = mdl.Coefficients.Estimate(5:8);
        p   = mdl.Coefficients.pValue(5:8);
        t   = mdl.Coefficients.tStat(5:8);
        SE  = mdl.Coefficients.SE(5:8);
    elseif strcmpi(nameofdataset,'second') 
        x   = mdl.Coefficients.Estimate(4:7);
        p   = mdl.Coefficients.pValue(4:7);
        t   = mdl.Coefficients.tStat(4:7);
        SE  = mdl.Coefficients.SE(4:7);
    end
    CI  = SE * 1.959964;

    fprintf('Regression (%s):\n',measstr);
    for ivar = 1:4 % ad, cit, sw, icar
        fprintf('%s: %+.2f ± %.2f; t=%+.4f ',pad(varstr{ivar},5),x(ivar),CI(ivar),t(ivar));
        if strcmpi(nameofdataset,'first') 
            fprintf(' (p=%.4f)',p(ivar)); % p-values for first dataset
        else
            %
            % see R code for replication Bayes Factor values on second dataset
            %
        end
        disp(' ')
    end
    % correlation CIT ~ parameters
    [r,p] = corr(y,dim_data(idx_incl,2),'Type','Spearman');
    fprintf('Correlation %s-%s: r=%+.2f, p=%.4f\n','CIT',measstr,r,p);

    % condition-wise correlation
    for icond = 1:3
        cond = condorder(icond);
        [r(cond),p] = corr(pars(idx_incl,ipar,cond),dim_data(idx_incl,2),'Type','Spearman');
        fprintf('Correlation %s-%s (%s): r=%+.2f, p=%.4f\n','CIT',measstr,pad(condstr{cond},3), ...
            r(cond),p);
    end
    disp(' ');
    if imeas == 2
        % Bayes factor test comparing how much more likely was it that the S+
        % correlation was observed under
        % H1: Distribution of V+
        % H0: Null distribution
    
        % perform Fisher z-transform on Spearman's rho values
        z_v = atanh(r(2)); % rho (V+)
        z_s = atanh(r(3)); % rho (S+)
        z_r = atanh(r(1));
        std = 1/sqrt(nsubj-3);
        
        p_0 = normpdf(z_s,0,std);   % H0: Observing S+ correlation under null distribution
        p_1 = normpdf(z_s,z_v,std); % H1: Observing S+ correlation under V+ distribution

        % Evidence that the correlation found in S+ is observed under H1 than H0
        fprintf('BF_V0 = %.2f\n',p_1/p_0);
        
        p_0 = normpdf(0,0,std);   % H0: Observing no correlation when there is none
        p_1 = normpdf(0,z_s,std); % H1: Observing no correlation in S+

        % Evidence of no observed correlation under a null
        % distribution than in the distribution of S+
        fprintf('BF_S0 = %.2f\n',p_0/p_1);
        

        p_0 = normpdf(z_s,z_v,std);   % H0: Observing no correlation when there is none
        p_1 = normpdf(z_s,z_r,std); % H1: Observing no correlation in S+

        % Evidence of no observed correlation under a null
        % distribution than in the distribution of S+
        fprintf('BF_VR = %.2f\n',p_0/p_1);

        disp(' ');
    end
end

%% Figure 6

%
% see script_fig6.m
%

%% Supplementary Figure 5
clc
for imeas = 1:4
    switch imeas
        case 1 % accuracy
        y = mean((pcor_raw(idx_incl,:)),2);
        measstr = 'accuracy';
        case 2 % switch rate
        y = mean((1-prep_raw(idx_incl,:)),2);
        measstr = 'switch rate';
        case 3 % learning noise
        y = mean((pars(idx_incl,3,:)),3);
        measstr = 'learning noise';
        case 4 % choice temperature
        y = mean((pars(idx_incl,4,:)),3);
        measstr = 'choice temperature';
    end
    y    = tiedrank(y);
    icar = tiedrank(ques_scores.iq(idx_incl))';
    age  = id_age_sex.age(idx_incl);
    sex  = id_age_sex.sex(idx_incl);

    fprintf('Regression (%s):\n',measstr);
    for iques = 1:numel(ques_names)-1
        scores = ques_scores.(ques_names{iques})(idx_incl)';
        scores = tiedrank(scores);

        X = table(scores,age,sex,icar,y);
        mdl = fitglm(X);
        x = mdl.Coefficients.Estimate(2); % coefficient for score
        t = mdl.Coefficients.tStat(2);
        p = mdl.Coefficients.pValue(2); % p=value for coefficient for score
        SE = mdl.Coefficients.SE(2); % SE on coeffs above
        CI = SE * 1.959964;
        fprintf(' %s: %+.2f ± %.2f; t = %+.4f,',pad(ques_names{iques},7,'right'),x,CI,t);
        
        if strcmpi(nameofdataset,'first') 
            fprintf(' (p=%.4f)\n',p); % p-values for first dataset
        else
            disp(' ')
            % 
            % T-values and degrees of freedom (i.e. number of participants)
            % are entered in the R code for replication Bayes Factor values 
            % on the second dataset (bayesFactorCalculations.R)
            %
        end
    end
    disp(' ');
end
