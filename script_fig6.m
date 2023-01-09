
%% Compute test-retest correlations

% clear workspace
clear all
close all
clc

addpath(genpath('./data'));

% set analysis parameters
use_logtr = true; % use log-transform?
nboot = 1e5; % number of bootstrap resamples

% load data
% all these intermediate files have been generated by prior scripts
% they can also be regenerated if necessary
load('parameter_fits_first.mat','pars');
load('parameter_fits_retest_first.mat','pars_rt');

% load('excl_cond_sample1.mat','excl_cond');
load('dimension_scores_icar_excl_first.mat','ques_data');
pcor_cond = load('pcorrect_average_first.mat','pcor_raw');
pcor_cond = pcor_cond.pcor_raw;
prep_cond = load('prepeat_average_first.mat','prep_raw');
prep_cond = prep_cond.prep_raw;
pcor_rt = load('pcorrect_average_retest_first.mat','pcor_raw');
pcor_rt = pcor_rt.pcor_raw;
prep_rt = load('prepeat_average_retest_first.mat','prep_raw');
prep_rt = prep_rt.prep_raw;

idx_ques = any(~isnan(ques_data(:,1:3)),2) & ~ques_data(:,5);
ncorrect = 90; % one lower will pass binomial test below
ntrialspercond = 160;
pcorrect_threshold = ncorrect/ntrialspercond;
excl_cond = pcor_cond < pcorrect_threshold | isnan(pcor_cond) & idx_ques;
excl_cond = excl_cond;

% reorder conditions as 1=Ref 2=Unp 3=Vol!
excl_cond = excl_cond(:,[1,3,2]);
pars      = pars(:,:,[1,3,2]);
pars_rt   = pars_rt(:,:,[1,3,2]);
pcor_cond = pcor_cond(:,[1,3,2]);
prep_cond = prep_cond(:,[1,3,2]);
pcor_rt   = pcor_rt(:,[1,3,2]);
prep_rt   = prep_rt(:,[1,3,2]);

% use log-transform
if use_logtr
    fprintf('Using log-transform.\n');
    pars(:,1:2,:) = log(pars(:,1:2,:)./(1-pars(:,1:2,:)));
    pars(:,3:4,:) = log(pars(:,3:4,:));
    pars_rt(:,1:2,:) = log(pars_rt(:,1:2,:)./(1-pars_rt(:,1:2,:)));
    pars_rt(:,3:4,:) = log(pars_rt(:,3:4,:));
    pcor_cond = log(pcor_cond./(1-pcor_cond));
    prep_cond = log(prep_cond./(1-prep_cond));
    pcor_rt = log(pcor_rt./(1-pcor_rt));
    prep_rt = log(prep_rt./(1-prep_rt));
end

% exclude bad/missing subjects
excl = any(excl_cond,2) | any(isnan(pars_rt),[2,3]);
xvar = pars(~excl,:,:);
yvar = pars_rt(~excl,:,:);
pcor = pcor_cond(~excl,:);
prep = prep_cond(~excl,:);
ques = ques_data(~excl,1:3);
pcor_rt = pcor_rt(~excl,:);
prep_rt = prep_rt(~excl,:);

nsubj = size(xvar,1);
fprintf('Found %d subjects to use.\n',nsubj);

% compute test-retest correlation for p(correct)
rho_trt = nan(nboot,1);
for iboot = 1:nboot
    isubj = randsample(nsubj,nsubj,true);
    rho_trt(iboot) = corr(mean(pcor(isubj,:),2),mean(pcor_rt(isubj,:),2),'type','pearson');
end
rho_trt_raw = corr(mean(pcor,2),mean(pcor_rt,2),'type','pearson');
save('./rho_trt_pcor.mat','rho_trt*');

% compute test-retest correlation for p(repeat)
rho_trt = nan(nboot,1);
for iboot = 1:nboot
    isubj = randsample(nsubj,nsubj,true);
    rho_trt(iboot) = corr(mean(prep(isubj,:),2),mean(prep_rt(isubj,:),2),'type','pearson');
end
rho_trt_raw = corr(mean(prep,2),mean(prep_rt,2),'type','pearson');
save('./rho_trt_prep.mat','rho_trt*');

% compute test-retest correlations for model parameters
for ipar = 1:4
    rho_trt = nan(nboot,1);
    for iboot = 1:nboot
        isubj = randsample(nsubj,nsubj,true);
        rho_trt(iboot) = corr(mean(xvar(isubj,ipar,:),3),mean(yvar(isubj,ipar,:),3),'type','pearson');
    end
    rho_trt_raw = corr(mean(xvar(:,ipar,:),3),mean(yvar(:,ipar,:),3),'type','pearson');
    save(sprintf('./rho_trt_par%d.mat',ipar),'rho_trt*');
end

fprintf('Done.\n\n');

%% Compute ICC for model parameter and plot test-retest scatter
%
%  Parameters:
%    ipar = 1 => alpha (learning rate)
%    ipar = 2 => delta (decay rate)
%    ipar = 3 => zeta (learning noise)
%    ipar = 4 => tau (choice temperature)
%
%  Warning: this cell needs previous cells to have been run.

% clear figures and command window
close all
clc

% set analysis parameters
ipar = 4; % model parameter index
icctype = 'A-1'; % ICC type

% compute ICC
icc_hat = nan(1,4);
icc_loc = nan(1,4);
icc_hic = nan(1,4);
pval    = nan(1,4);
for icond = 1:3
    [icc_hat(icond),icc_loc(icond),icc_hic(icond),~,~,~,pval(icond)] = ICC([xvar(:,ipar,icond),yvar(:,ipar,icond)],icctype);
end
[icc_hat(4),icc_loc(4),icc_hic(4),~,~,~,pval(4)] = ICC([mean(xvar(:,ipar,:),3),mean(yvar(:,ipar,:),3)],icctype);

% compute ICC CI-95%
icc_hat = icc_hat([2,1,3,4]); % S+/Ref/V+/all
icc_loc = icc_loc([2,1,3,4]); % S+/Ref/V+/all
icc_hic = icc_hic([2,1,3,4]); % S+/Ref/V+/all

% set plot parameters
pbar = 1; % plot box aspect ratio (width/height)
figh = 4; % figure height (cm)

% plot ICC
hf = figure('Color','white');
hold on
xlim([-0.4,3.4]);
ylim([-0.2,1]);
for i = 1:3
    bar(i,icc_hat(i),0.8);
    plot([i,i],[icc_loc(i),icc_hic(i)],'-','LineWidth',1);
end
plot(xlim,icc_hat(4)*[1,1],'k-');
plot(0,icc_hat(4),'o','Color','none','MarkerFaceColor','k','MarkerSize',5);
plot([0,0],[icc_loc(4),icc_hic(4)],'k-','LineWidth',1);
hold off
set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1),'LineWidth',0.75);
set(gca,'FontName','Helvetica','FontSize',7.2);
set(gca,'XTick',1:3,'XTickLabel',{'S+','Ref','V+'});
set(gca,'YTick',0:0.2:1);
xlabel('condition','FontSize',8);
ylabel('ICC','FontSize',8);
axes = findobj(gcf, 'type', 'axes');
for a = 1:length(axes),
    if axes(a).YColor <= [1 1 1]
        axes(a).YColor = [0 0 0];
    end
    if axes(a).XColor <= [1 1 1]
        axes(a).XColor = [0 0 0];
    end
end
if ~isempty(figh)
    set(hf,'PaperPositionMode','manual', ...
        'PaperPosition',[2.5,13,16,figh],'PaperUnits','centimeters', ...
        'PaperType','A4','PaperOrientation','portrait');
    figure(hf);
    fname = './figs/fig_par';
%     print(fname,'-painters','-dpdf');
end

% set plot parameters
figh = 4; % figure height (cm)
pbar = 1; % plot box aspect ratio (width/height)

% compute intraclass correlation
[icc_hat,icc_loc,icc_hic] = ICC([mean(xvar(:,ipar,:),3),mean(yvar(:,ipar,:),3)],'1-1');
fprintf('ICC for par%d = %.3f [%.3f %.3f]\n',ipar,icc_hat,icc_loc,icc_hic);

% prepare additional plot parameters
switch ipar
    case 3
        xyl = [-1.5,+1];
        rgb = [109,207,244]/255;
        xt = log([0.5,1.0,2.0]);
        xts = {'0.5','1.0','2.0'};
        yt = log([0.5,1.0,2.0]);
        yts = {'0.5','1.0','2.0'};
    case 4
        xyl = [-5.5,-2];
        rgb = [186,126,180]/255;
        xt = log([0.01,0.02,0.05,0.1]);
        xts = {'0.01','0.02','0.05','0.1'};
        yt = log([0.01,0.02,0.05,0.1]);
        yts = {'0.01','0.02','0.05','0.1'};
    otherwise
        error('Undefined model parameter index!');
end
xs = 'test';
ys = 'retest';

% plot test-retest scatter
hf = figure('Color','white');
hold on
xlim(xyl);
ylim(xyl);
plot(xlim,ylim,'k-');
mu = [mean(mean(xvar(:,ipar,:),3)),mean(mean(yvar(:,ipar,:),3))]; % means
s2 = cov([mean(xvar(:,ipar,:),3),mean(yvar(:,ipar,:),3)]); % covariance matrix
xv = xyl(1):0.01:xyl(2);
[x1,x2] = ndgrid(xv,xv);
x = [x1(:),x2(:)];
p = mvnpdf(x,mu,s2);
p = reshape(p,size(x1))';
contour(xv,xv,p,max(p(:))*[0.25,0.5,0.75],'Color',0.5*(rgb+1));
c = contour(xv,xv,p,max(p(:))*[0.25,0.25],'Color',0.5*(rgb+1));
patch(c(1,:),c(2,:),0.5*(rgb+1),'EdgeColor','none','FaceAlpha',0.25/4);
c = contour(xv,xv,p,max(p(:))*[0.5,0.5],'Color',0.5*(rgb+1));
patch(c(1,:),c(2,:),0.5*(rgb+1),'EdgeColor','none','FaceAlpha',0.5/4);
c = contour(xv,xv,p,max(p(:))*[0.75,0.75],'Color',0.5*(rgb+1));
patch(c(1,:),c(2,:),0.5*(rgb+1),'EdgeColor','none','FaceAlpha',0.75/4);
p = scatter(mean(xvar(:,ipar,:),3),mean(yvar(:,ipar,:),3),8, ...
    'MarkerFaceColor',rgb,'MarkerEdgeColor','none');
p.MarkerFaceAlpha = 0.5;
hold off
set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1));
set(gca,'FontName','Helvetica','FontSize',7.2);
set(gca,'XTick',xt,'XTickLabel',xts);
set(gca,'YTick',yt,'YTickLabel',yts);
xlabel(xs,'FontSize',8);
ylabel(ys,'FontSize',8);
axes = findobj(gcf, 'type', 'axes');
for a = 1:length(axes),
    if axes(a).YColor <= [1 1 1]
        axes(a).YColor = [0 0 0];
    end
    if axes(a).XColor <= [1 1 1]
        axes(a).XColor = [0 0 0];
    end
end
if ~isempty(figh)
    set(hf,'PaperPositionMode','manual', ...
        'PaperPosition',[2.5,13,16,figh],'PaperUnits','centimeters', ...
        'PaperType','A4','PaperOrientation','portrait');
    figure(hf);
    fname = './figs/fig_par';
%     print(fname,'-painters','-dpdf');
end

%% Compute ICC and plot test-retest scatter for accuracy
%
%  Warning: this cell requires previous cells to have been run.

% clear figures and command window
close all
clc

% set analysis parameters
icctype = 'A-1'; % ICC type

% compute ICC
icc_hat = nan(1,4);
icc_loc = nan(1,4);
icc_hic = nan(1,4);
pval    = nan(1,4);
for icond = 1:3
    [icc_hat(icond),icc_loc(icond),icc_hic(icond),~,~,~,pval(icond)] = ICC([pcor(:,icond),pcor_rt(:,icond)],icctype);
end
[icc_hat(4),icc_loc(4),icc_hic(4),~,~,~,pval(4)] = ICC([mean(pcor,2),mean(pcor_rt,2)],icctype);

% compute ICC CI-95%
icc_hat = icc_hat([2,1,3,4]); % S+/Ref/V+/all
icc_loc = icc_loc([2,1,3,4]); % S+/Ref/V+/all
icc_hic = icc_hic([2,1,3,4]); % S+/Ref/V+/all

% set plot parameters
pbar = 1; % plot box aspect ratio (width/height)
figh = 4; % figure height (cm)

% plot ICC per condition and across all conditions
hf = figure('Color','white');
hold on
xlim([-0.4,3.4]);
ylim([-0.2,1]);
for i = 1:3
    bar(i,icc_hat(i),0.8);
    plot([i,i],[icc_loc(i),icc_hic(i)],'-','LineWidth',1);
end
plot(xlim,icc_hat(4)*[1,1],'k-');
plot(0,icc_hat(4),'o','Color','none','MarkerFaceColor','k','MarkerSize',5);
plot([0,0],[icc_loc(4),icc_hic(4)],'k-','LineWidth',1);
hold off
set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1),'LineWidth',0.75);
set(gca,'FontName','Helvetica','FontSize',7.2);
set(gca,'XTick',1:3,'XTickLabel',{'S+','Ref','V+'});
set(gca,'YTick',0:0.2:1);
xlabel('condition','FontSize',8);
ylabel('ICC','FontSize',8);
axes = findobj(gcf, 'type', 'axes');
for a = 1:length(axes),
    if axes(a).YColor <= [1 1 1]
        axes(a).YColor = [0 0 0];
    end
    if axes(a).XColor <= [1 1 1]
        axes(a).XColor = [0 0 0];
    end
end
if ~isempty(figh)
    set(hf,'PaperPositionMode','manual', ...
        'PaperPosition',[2.5,13,16,figh],'PaperUnits','centimeters', ...
        'PaperType','A4','PaperOrientation','portrait');
    figure(hf);
    fname = './figs/fig_par';
%     print(fname,'-painters','-dpdf');
end

% set plot parameters
xyl = [0,3.5];
rgb = [0.5,0.5,0.5];
xt = 0.5:0.1:1;
xts = {'0.5','0.6','0.7','0.8','0.9','1'};
yt = 0.5:0.1:1;
yts = {'0.5','0.6','0.7','0.8','0.9','1'};

% plot test-retest scatter for accuracy
hf = figure('Color','white');
hold on
xlim([0.5,1]);
ylim([0.5,1]);
plot(xlim,ylim,'k-');
mu = [mean(mean(pcor,2)),mean(mean(pcor_rt,2))]; % means
s2 = cov([mean(pcor,2),mean(pcor_rt,2)]); % covariance matrix
xv = xyl(1):0.01:xyl(2);
[x1,x2] = ndgrid(xv,xv);
x = [x1(:),x2(:)];
p = mvnpdf(x,mu,s2);
p = reshape(p,size(x1))';
xv = 1./(1+exp(-xv));
contour(xv,xv,p,max(p(:))*[0.25,0.5,0.75],'Color',0.5*(rgb+1));
c = contour(xv,xv,p,max(p(:))*[0.25,0.25],'Color',0.5*(rgb+1));
patch(c(1,:),c(2,:),0.5*(rgb+1),'EdgeColor','none','FaceAlpha',0.25/4);
c = contour(xv,xv,p,max(p(:))*[0.5,0.5],'Color',0.5*(rgb+1));
patch(c(1,:),c(2,:),0.5*(rgb+1),'EdgeColor','none','FaceAlpha',0.5/4);
c = contour(xv,xv,p,max(p(:))*[0.75,0.75],'Color',0.5*(rgb+1));
patch(c(1,:),c(2,:),0.5*(rgb+1),'EdgeColor','none','FaceAlpha',0.75/4);
itf = @(x)1./(1+exp(-x));
p = scatter(itf(mean(pcor,2)),itf(mean(pcor_rt,2)),8, ...
    'MarkerFaceColor',rgb,'MarkerEdgeColor','none');
p.MarkerFaceAlpha = 0.5;
hold off
set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1));
set(gca,'FontName','Helvetica','FontSize',7.2);
set(gca,'XTick',xt,'XTickLabel',xts);
set(gca,'YTick',yt,'YTickLabel',yts);
xlabel(xs,'FontSize',8);
ylabel(ys,'FontSize',8);
axes = findobj(gcf, 'type', 'axes');
for a = 1:length(axes),
    if axes(a).YColor <= [1 1 1]
        axes(a).YColor = [0 0 0];
    end
    if axes(a).XColor <= [1 1 1]
        axes(a).XColor = [0 0 0];
    end
end
if ~isempty(figh)
    set(hf,'PaperPositionMode','manual', ...
        'PaperPosition',[2.5,13,16,figh],'PaperUnits','centimeters', ...
        'PaperType','A4','PaperOrientation','portrait');
    figure(hf);
%     fname = './figs/fig_par';
%     print(fname,'-painters','-dpdf');
end

%% Compute ICC and plot test-retest scatter for switch rate
%
%  Warning: this cell requires previous cells to have been run.

% clear figures and command window
close all
clc

% set analysis parameters
icctype = 'A-1'; % ICC type

% compute ICC
icc_hat = nan(1,4);
icc_loc = nan(1,4);
icc_hic = nan(1,4);
pval    = nan(1,4);
for icond = 1:3
    [icc_hat(icond),icc_loc(icond),icc_hic(icond),~,~,~,pval(icond)] = ICC([prep(:,icond),prep_rt(:,icond)],icctype);
end
[icc_hat(4),icc_loc(4),icc_hic(4),~,~,~,pval(4)] = ICC([mean(prep,2),mean(prep_rt,2)],icctype);

% compute ICC CI-95%
icc_hat = icc_hat([2,1,3,4]); % S+/Ref/V+/all
icc_loc = icc_loc([2,1,3,4]); % S+/Ref/V+/all
icc_hic = icc_hic([2,1,3,4]); % S+/Ref/V+/all

% set plot parameters
pbar = 1; % plot box aspect ratio (width/height)
figh = 4; % figure height (cm)

% plot ICC per condition and across all conditions
hf = figure('Color','white');
hold on
xlim([-0.4,3.4]);
ylim([-0.2,1]);
for i = 1:3
    bar(i,icc_hat(i),0.8);
    plot([i,i],[icc_loc(i),icc_hic(i)],'-','LineWidth',1);
end
plot(xlim,icc_hat(4)*[1,1],'k-');
plot(0,icc_hat(4),'o','Color','none','MarkerFaceColor','k','MarkerSize',5);
plot([0,0],[icc_loc(4),icc_hic(4)],'k-','LineWidth',1);
hold off
set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1),'LineWidth',0.75);
set(gca,'FontName','Helvetica','FontSize',7.2);
set(gca,'XTick',1:3,'XTickLabel',{'S+','Ref','V+'});
set(gca,'YTick',0:0.2:1);
xlabel('condition','FontSize',8);
ylabel('ICC','FontSize',8);
axes = findobj(gcf, 'type', 'axes');
for a = 1:length(axes),
    if axes(a).YColor <= [1 1 1]
        axes(a).YColor = [0 0 0];
    end
    if axes(a).XColor <= [1 1 1]
        axes(a).XColor = [0 0 0];
    end
end
if ~isempty(figh)
    set(hf,'PaperPositionMode','manual', ...
        'PaperPosition',[2.5,13,16,figh],'PaperUnits','centimeters', ...
        'PaperType','A4','PaperOrientation','portrait');
    figure(hf);
%     fname = './figs/fig_par';
%     print(fname,'-painters','-dpdf');
end

% set plot parameters
xyl = [0,3.5];
rgb = [0.5,0.5,0.5];
xt = 0:0.1:0.5;
xts = {'0','0.1','0.2','0.3','0.4','0.5'};
yt = 0:0.1:0.5;
yts = {'0','0.1','0.2','0.3','0.4','0.5'};

% plot test-retest scatter for switch rate
hf = figure('Color','white');
hold on
xlim([0,0.5]);
ylim([0,0.5]);
plot(xlim,ylim,'k-');
mu = [mean(mean(prep,2)),mean(mean(prep_rt,2))]; % means
s2 = cov([mean(prep,2),mean(prep_rt,2)]); % covariance matrix
xv = xyl(1):0.01:xyl(2);
[x1,x2] = ndgrid(xv,xv);
x = [x1(:),x2(:)];
p = mvnpdf(x,mu,s2);
p = reshape(p,size(x1))';
xv = 1./(1+exp(-xv));
xv = 1-xv;
contour(xv,xv,p,max(p(:))*[0.25,0.5,0.75],'Color',0.5*(rgb+1));
c = contour(xv,xv,p,max(p(:))*[0.25,0.25],'Color',0.5*(rgb+1));
patch(c(1,:),c(2,:),0.5*(rgb+1),'EdgeColor','none','FaceAlpha',0.25/4);
c = contour(xv,xv,p,max(p(:))*[0.5,0.5],'Color',0.5*(rgb+1));
patch(c(1,:),c(2,:),0.5*(rgb+1),'EdgeColor','none','FaceAlpha',0.5/4);
c = contour(xv,xv,p,max(p(:))*[0.75,0.75],'Color',0.5*(rgb+1));
patch(c(1,:),c(2,:),0.5*(rgb+1),'EdgeColor','none','FaceAlpha',0.75/4);
itf = @(x)1-1./(1+exp(-x));
p = scatter(itf(mean(prep,2)),itf(mean(prep_rt,2)),8, ...
    'MarkerFaceColor',rgb,'MarkerEdgeColor','none');
p.MarkerFaceAlpha = 0.5;
hold off
set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1));
set(gca,'FontName','Helvetica','FontSize',7.2);
set(gca,'XTick',xt,'XTickLabel',xts);
set(gca,'YTick',yt,'YTickLabel',yts);
xlabel(xs,'FontSize',8);
ylabel(ys,'FontSize',8);
axes = findobj(gcf, 'type', 'axes');
for a = 1:length(axes),
    if axes(a).YColor <= [1 1 1]
        axes(a).YColor = [0 0 0];
    end
    if axes(a).XColor <= [1 1 1]
        axes(a).XColor = [0 0 0];
    end
end
if ~isempty(figh)
    set(hf,'PaperPositionMode','manual', ...
        'PaperPosition',[2.5,13,16,figh],'PaperUnits','centimeters', ...
        'PaperType','A4','PaperOrientation','portrait');
    figure(hf);
%     fname = './figs/fig_par';
%     print(fname,'-painters','-dpdf');
end

%% Compute CIT correlations and run mediation analysis
%
%  This cell requires the MediationToolbox and its dependencies:
%    https://github.com/canlab/MediationToolbox
%  Download this toolbox into the directory /toolbox/
%
%  This cell runs the analyses but does not plot the results.

% clear workspace
clear all
close all
clc

% add MediationToolbox and dependent folders to path
addpath('./toolbox/MediationToolbox/mediation_toolbox/');
addpath('./toolbox/MediationToolbox/mediation_toolbox/Boot_samples_needed_fcns/');
addpath('./toolbox/MediationToolbox/geom2d/');

% set analysis parameters
isample = [1]; % sample number
use_logtr = true; % use log-transform?
nboot = 1e5; % number of bootstrap resamples

nsample = numel(isample);
if nsample == 1
    % load data from single sample
    load(sprintf('../out/pars_sample%d.mat',isample),'pars');
    load(sprintf('../out/excl_cond_sample%d.mat',isample),'excl_cond');
    load(sprintf('../out/psych_dim_icar_sample%d.mat',isample),'ques_data');
    load(sprintf('../out/pcor_sample%d.mat',isample),'pcor_cond');
    load(sprintf('../out/prep_sample%d.mat',isample),'prep_cond');
else
    % load data from multiple samples
    pars_tmp = [];
    excl_cond_tmp = [];
    ques_data_tmp = [];
    pcor_cond_tmp = [];
    prep_cond_tmp = [];
    for i = 1:nsample
        load(sprintf('../out/pars_sample%d.mat',isample(i)),'pars');
        load(sprintf('../out/excl_cond_sample%d.mat',isample(i)),'excl_cond');
        load(sprintf('../out/psych_dim_icar_sample%d.mat',isample(i)),'ques_data');
        load(sprintf('../out/pcor_sample%d.mat',isample(i)),'pcor_cond');
        load(sprintf('../out/prep_sample%d.mat',isample(i)),'prep_cond');
        pars_tmp = cat(1,pars_tmp,pars);
        excl_cond_tmp = cat(1,excl_cond_tmp,excl_cond);
        ques_data_tmp = cat(1,ques_data_tmp,ques_data);
        pcor_cond_tmp = cat(1,pcor_cond_tmp,pcor_cond);
        prep_cond_tmp = cat(1,prep_cond_tmp,prep_cond);
    end
    pars = pars_tmp;
    excl_cond = excl_cond_tmp;
    ques_data = ques_data_tmp;
    pcor_cond = pcor_cond_tmp;
    prep_cond = prep_cond_tmp;
end

% reorder conditions as 1=Ref 2=Unp 3=Vol
pars      = pars(:,:,[1,3,2]);
excl_cond = excl_cond(:,[1,3,2]);
pcor_cond = pcor_cond(:,[1,3,2]);
prep_cond = prep_cond(:,[1,3,2]);

% use log-transform
if use_logtr
    fprintf('Using log-transform.\n');
    pars(:,1:2,:) = log(pars(:,1:2,:)./(1-pars(:,1:2,:)));
    pars(:,3:4,:) = log(pars(:,3:4,:));
    pcor_cond = log(pcor_cond./(1-pcor_cond));
    prep_cond = log(prep_cond./(1-prep_cond));
end

% exclude bad/missing subjects
excl = any(excl_cond,2) | any(isnan(ques_data),2);
xvar = pars(~excl,:,:);
pcor = pcor_cond(~excl,:);
prep = prep_cond(~excl,:);
ques = ques_data(~excl,1:3);

nsubj = size(xvar,1);
fprintf('Found %d subjects to use.\n',nsubj);

nb = 1e4;
pb = [];
for ib = 1:nb
    idat = randsample(nsubj,nsubj,true);
    pb(ib,:) = mediation(ques(idat,2),mean(xvar(idat,4,:),3),-mean(prep(idat,:),2));
end
[paths,stats] = mediation(ques(:,2),mean(xvar(:,4,:),3),-mean(prep,2), 'plots', 'verbose');

% compute CIT correlation with p(correct)
rho_cit = nan(nboot,1);
for iboot = 1:nboot
    isubj = randsample(nsubj,nsubj,true);
    rho_cit(iboot) = corr(ques(isubj,2),mean(pcor(isubj,:),2),'type','pearson');
end
fprintf('rho[CIT] for p(correct) = %+.3f\n',mean(rho_cit));
rho_cit_raw = corr(ques(:,2),mean(pcor,2),'type','pearson');
if nsample == 1
    save(sprintf('./rho_cit_pcor_sample%d.mat',isample),'rho_cit*');
else
    save('./rho_cit_pcor_samples.mat','rho_cit*');
end

% compute CIT correlation with p(repeat)
rho_cit = nan(nboot,1);
for iboot = 1:nboot
    isubj = randsample(nsubj,nsubj,true);
    rho_cit(iboot) = corr(ques(isubj,2),mean(prep(isubj,:),2),'type','pearson');
end
fprintf('rho[CIT] for p(repeat) = %+.3f\n',mean(rho_cit));
rho_cit_raw = corr(ques(:,2),mean(prep,2),'type','pearson');
if nsample == 1
    save(sprintf('./rho_cit_prep_sample%d.mat',isample),'rho_cit*');
else
    save('./rho_cit_prep_samples.mat','rho_cit*');
end

% compute CIT correlations with model parameters
for ipar = 1:4
    rho_cit = nan(nboot,1);
    for iboot = 1:nboot
        isubj = randsample(nsubj,nsubj,true);
        rho_cit(iboot) = corr(ques(isubj,2),mean(xvar(isubj,ipar,:),3),'type','pearson');
    end
    fprintf('rho[CIT] for par%d = %+.3f\n',ipar,mean(rho_cit));
    rho_cit_raw = corr(ques(:,2),mean(xvar(:,ipar,:),3),'type','pearson');
    if nsample == 1
        save(sprintf('./rho_cit_par%d_sample%d.mat',ipar,isample),'rho_cit*');
    else
        save(sprintf('./rho_cit_par%d_samples.mat',ipar),'rho_cit*');
    end
end

fprintf('Done.\n\n');

%% Compute fraction trait explained by CIT
%
%  Warning: previous cells need to be run before this one!

% clear workspace, figures and command window
clear all
close all
clc

% load requested files
% par1 = alpha (learning rate)
% par2 = delta (decay rate)
% par3 = zeta (learning noise)
% par4 = tau (choice temperature)
load ./rho_trt_par3.mat
load ./rho_cit_par3_sample1.mat

% set plot parameters
figh = 4; % figure height (cm)
pbar = 1; % plot box aspect ratio (width/height)
rgb  = [0.5,0.5,0.5]; % color

% plot bootstrap distribution
hf = figure('Color','white');
hold on
xlim([0,1]);
histogram(rho_cit.^2./rho_trt.^2,0:0.04:1,'Normalization','pdf','EdgeColor','w','FaceColor',0.5*(rgb+1));
[pk,xk] = ksdensity(rho_cit.^2./rho_trt.^2,0:0.02:1);
plot(xk,pk,'-','LineWidth',2,'Color',rgb);
ylim([0,12]);
plot(quantile(rho_cit.^2./rho_trt.^2,[0.025,0.975]),max(pk)*1.1*[1,1],'-','Color',rgb,'LineWidth',1);
plot(median(rho_cit.^2./rho_trt.^2),max(pk)*1.1,'o','Color','w','MarkerFaceColor',rgb,'MarkerSize',5);
hold off
set(gca,'Layer','top','Box','off','PlotBoxAspectRatio',[pbar,1,1]);
set(gca,'TickDir','out','TickLength',[1,1]*0.02/max(pbar,1),'LineWidth',0.75);
set(gca,'FontName','Helvetica','FontSize',7.2);
set(gca,'XTick',0:0.2:1);
set(gca,'YTick',0);
xlabel('fraction trait explained by CIT','FontSize',8);
ylabel('pdf','FontSize',8);
axes = findobj(gcf, 'type', 'axes');
for a = 1:length(axes),
    if axes(a).YColor <= [1 1 1]
        axes(a).YColor = [0 0 0];
    end
    if axes(a).XColor <= [1 1 1]
        axes(a).XColor = [0 0 0];
    end
end
if ~isempty(figh)
    set(hf,'PaperPositionMode','manual', ...
        'PaperPosition',[2.5,13,16,figh],'PaperUnits','centimeters', ...
        'PaperType','A4','PaperOrientation','portrait');
    figure(hf);
    fname = './figs/fig_hist';
%     print(fname,'-painters','-dpdf');
end

% compute metric for both the raw and bootstrap-resampled data
rho_cit_raw.^2./rho_trt_raw.^2
median(rho_cit.^2./rho_trt.^2)
