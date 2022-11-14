ccc;
israin=1;
load('fluxdata.mat', 'lai_site','lai_date'); % extracted LAI from MODIS
lai_date   = datenum(lai_date(:,1),lai_date(:,2),lai_date(:,3));

% samples from all available flux sites
FPAR_ML=[]; 
SM_ML=[];
VPD_ML=[];
PFT_ML=[];
TA_ML=[];
CA_ML=[];
RN_ML=[];
F_ML=[];
LE_ML=[];
WS_ML=[];
PA_ML=[];
RH_ML=[];
GPP_ML=[];
LAI_ML=[];
CP_ML=[];
for kk=1:6
    eval(['P_at',num2str(kk),'_ML=[];']);  % P_at1_ML=[];
    eval(['W_at',num2str(kk),'_ML=[];']);  % W_at1_ML=[];
end
P_at12_ML=[];
P_at24_ML=[];
P_at48_ML=[];
G_ML=[];
% FPAR_site_ML=[];
PPFD_IN_ML=[];
% PPFD_OUT_ML=[];
h_measure_ML=[];
h_canopy_ML=[];
P_ML=[];
Timescale_ML=[];

% for the reproduce of the oriiginal data
P_id_org=[];
% P_id_a_org=[];
% P_id_n_org=[];
P_org=[];
CP_org=[];
Rn_org=[];
SWC_org=[];
Ta_org=[];
WS_org=[];
LEclose_org=[];
FN_org=[];
removeind=[];

GS_ML      =[];
Rs_ML      =[];    % Rs surface resistance using LE s/m
Ra_ML      =[];
Delta_ML   =[];
Density_ML =[];
r_ML       =[];
USTAR_ML   =[];
H_ML       =[];
LEclose_ML =[];    % Latent heat flux after corrected by Bowen ratio closure W/m-2
BR_ML      =[];
Rscorr_ML  =[];    % target variable: Rscorr surface resistance using LEclose  s/m
Gscorr_ML  =[];    % Gscorr stomatal conductance using Rscorr m/s

soil_property_WRB_ML  = [];
soil_property_USDA_ML = [];
DateNumber_ML = [];    % Date year-month-day
SiteNumber_ML = []; 
Pid_ML=[];
FD_ML=[];
FN_ML=[];
DN_ML=[];
SN_ML=[];
date1_ML=[];


data=dir('MAT_DATA_hourly\*.mat');
% Fluxnet_info = readtable('Fluxnet_sites_Hysteresis.csv');
Fluxnet_info = readtable('Fluxnet_sites_Hysteresis.csv');
k=1;sn=0; % k is hte number of rain events
for i=1:length(data)
    if strcmpi(data(i).name(1),'.')
        continue
    end
    
    load(fullfile('./MAT_DATA_hourly',data(i).name))
    disp(['i=',num2str(i),',',data(i).name(1:6),'; '])

    % obtain r % psychrometric constant, kPa/C
    r          = 10^(-6)*Cp*PA./(0.622*(2.501-0.00236*TA));
    
    % LEcorr Bowen Ratio closure
    LEclose = LE;
    LEcorr = (NETRAD-G)./(1+(H./LE));
    BR = (H+LE)./(NETRAD-G);
    LEclose=LEcorr;
    LEclose(LEclose<0)=0;

    % Rscorr 
    Lv = 2.5 * 10^6;
    Rv = 461;
    Cp = 1012;
    es = Delta.*TA.^2.*(Rv/Lv);
    para_1 = (Lv/Rv).*(Delta.*TA.^(-2) - 2*es.*TA.^(-3));    
    if all(isnan(G))        
%         Rscorr        =  (Delta.*(NETRAD-LEcorr) + Cp*Density.*VPD./Ra)./(r.*LEcorr./Ra)-Ra;
%         Rscorr        =  (Delta.*(NETRAD-LEclose) + Cp*Density.*VPD./Ra)./(r.*LEclose./Ra)-Ra;
        % Density: density of air, kg/m3
        Rscorr = (Density.*1012./(r.*LEclose)).*((Delta.*Ra.*(NETRAD-LEclose)./(Density.*1012))+(1/2).*para_1.*(Ra./(Density.*Cp)).^2.*(NETRAD-LEclose).^2+VPD)-Ra;
    else        
%         Rscorr        = (Delta.*(NETRAD-G-LEcorr) + Cp*Density.*VPD./Ra)./(r.*LEcorr./Ra)-Ra;
%         Rscorr        = (Delta.*(NETRAD-G-LEclose) + Cp*Density.*VPD./Ra)./(r.*LEclose./Ra)-Ra;
        Rscorr = (Density.*1012./(r.*LEclose)).*((Delta.*Ra.*(NETRAD-G-LEclose)./(Density.*1012))+(1/2).*para_1.*(Ra./(Density.*Cp)).^2.*(NETRAD-G-LEclose).^2+VPD)-Ra;
    end
    % obtain Gscorr using Rscorr 
    %  constraint
    Gscorr = 1000*1./Rscorr;  % unit: mm/s

    % VPD_leaf !
    VPD_leaf              = (1000./Gscorr).*(LEclose.*r)./(Density*Cp);  % unit: kPa
    VPD_leaf(VPD_leaf<0)  = NaN;
    VPD_leaf(VPD_leaf>2*max(VPD)) = NaN;
    Gscorr = Gscorr.*PA./(8.314*(273+TA));  % unit transformation: mm/s --> mol/m2/s

%     processing outliers of Gscorr
    Gscorr = REMOVE_OUTLIER(Gscorr);

    % remove negative values
    Gscorr(Gscorr<=0)=nan;
    
    sitecode   =    data(i).name(1:6);
    
    ind = findstritem(Fluxnet_info.Site_name,sitecode);
    if ~isempty(ind)
        number       = Fluxnet_info.No(ind);
        cover_type   = Fluxnet_info.Cover_type{ind};
        h.canopy     = Fluxnet_info.Canopy_hnew(ind);
        h.measure    = Fluxnet_info.Tower_hnew(ind);
        latitude     = Fluxnet_info.Latitude(ind);
        longitude    = Fluxnet_info.Longitude(ind);
        elevation    = Fluxnet_info.Elevation(ind);
        CP           = Fluxnet_info.CP_new(ind);
        lairatio     = Fluxnet_info.LAI_ratio(ind);
        if isnan(lairatio)
            lairatio=1;
        end
%         soil_property_WRB = Fluxnet_info.TAXNWRB(ind);
%         soil_property_USDA = Fluxnet_info.TAXOUSDA(ind);
    else
        continue;
    end

    if strcmp(cover_type,'WSA')
        cover_type='SAV';
    elseif strcmp(cover_type,'OSH')||strcmp(cover_type,'CSH')
        cover_type='SHU';
    end
        
    IGBP={'ENF','EBF','DNF','DBF','MF','SHU',...
        'SAV','GRA','WET','CRO'};
    % Woody savanna to savanna; CSH and OSH to SHU
    pftind=findstritem(IGBP,cover_type);
    PFT=pftind*ones(length(day),1);
    
    h_measure = h.measure*ones(length(day),1);
    h_canopy  = h.canopy*ones(length(day),1);
    CP  = CP*ones(length(day),1);
    Timescale = file_flag*ones(length(day),1);
    SN        = sn+1:1:sn+length(day);
    SN = SN';
    sn=sn+length(P);
%     soil_property_WRB  = soil_property_WRB * ones(length(day),1);
%     soil_property_USDA = soil_property_USDA * ones(length(day),1);

    P_id      = zeros(length(day),1);
    FN        = zeros(length(day),1);
    PreNumber = (1:1:length(day))';
    P_date    = unique(PreNumber(P>0.5)); % r
    P_date1   = P_date;
    
    % calculate the sunrise and sunset times
%     days = datenum(2017,1,1:365);
    [srise,sset,~]=sunrise(latitude,longitude,0,round(longitude/15),DateNumber);
    [~,~,~,H,MN,~] = datevec(srise);
    srise=H+MN/60;
    [~,~,~,H,MN,~] = datevec(sset);
    sset=H+MN/60;
    
    isinday=false(length(P),1);
    isinday((hours>srise)&(hours<sset))=1;

    P_date1_new=P_date1;
    for ni=1:length(P_date1)
        if isinday(ni)
            % if during daytime: consider the next 06 hours
            P_date1_new=cat(1,P_date1_new,((P_date(ni)+1):(P_date(ni)+6))');
        else
            % if during nighttime: consider the next 12 hours
            P_date1_new=cat(1,P_date1_new,((P_date(ni)+1):(P_date(ni)+12))');
        end
    end
    P_date1    = unique(P_date1_new);
    clearvars P_date1_new

%     for ni=1:12  % 之后12个小时也考虑在内
%         P_date1    = [P_date1;P_date+ni];
%         P_date1    = unique(P_date1);
%     end
    
    P_d    = unique(P_date1);  % indx of rainy days (including +12h)
    P_d(P_d>length(P))=[];
    P_d2   = [P_d(2:end);P_d(end)+1]; % indx of rain day +1
    P_d3   = P_d2-P_d;
    P_d4   = find(P_d3~=1);  % no rain dates
    P_d4=[0;P_d4;length(P_d3)]; % start indx of each rain event
    
    for j=1:length(P_d4)-1
        nn=P_d(P_d4(j)+1:P_d4(j+1));
        
        
        kn=find(P(nn)>1); % Check
        knn=kn(2:end)-kn(1:end-1); % 超过1mm事件的持续时长
        knnn=find(knn>=6);
        if (length(nn)>60)&&(~isempty(knnn)) % 总时长超过60小时且有1mm事件持续超过6小时的
            knnn=[1;knnn+1];
            for l=1:length(knnn)-1
                P_id(nn(kn(knnn(l)):kn(knnn(l+1))-1))=k;
                FN(nn(kn(knnn(l)):kn(knnn(l+1))-1))=1:length(nn(kn(knnn(l)):kn(knnn(l+1))-1));
                %                 P_id(nn())=k;
                P_id_n(k)=length(nn(kn(knnn(l)):kn(knnn(l+1))-1));
                P_id_a(k)=sum(P(nn(kn(knnn(l)):kn(knnn(l+1))-1)));
                P_id_max(k)=max(P(nn(kn(knnn(l)):kn(knnn(l+1))-1)));
                P_id_s(k)=submat(hours(nn(kn(knnn(l)):kn(knnn(l+1))-1)),1);
                Pt=P(nn(kn(knnn(l)):kn(knnn(l+1))-1));
                Wt=WS(nn(kn(knnn(l)):kn(knnn(l+1))-1));
                WSmax(k)=mean(Wt(Pt==max(Pt)));
%                 fpart=fpar(nn(kn(knnn(l)):kn(knnn(l+1))-1));
%                 fpmax(k)=mean(fpar(Pt==max(Pt)));
                k=k+1;
            end
            P_id(nn(kn(knnn(end)):end))=k;
            FN(nn(kn(knnn(end)):end))=1:length(nn(kn(knnn(end)):end));
            P_id_n(k)=length(nn(kn(knnn(end)):end));
            P_id_a(k)=sum(P(nn(kn(knnn(end)):end)));
            P_id_s(k)=submat(hours(nn(kn(knnn(end)):end)),1);
            P_id_max(k)=max(P(nn(kn(knnn(end)):end)));
            Pt=P(nn(kn(knnn(end)):end));
            Wt=WS(nn(kn(knnn(end)):end));
            WSmax(k)=mean(Wt(Pt==max(Pt)));
%             fpart=fpar(nn(kn(knnn(end)):end));
%             fpmax(k)=mean(fpar(Pt==max(Pt)));
            k=k+1;
        else
            % P_id2 is the global index of rain events (从当前事件开始到下一次事件开始的index)
            % FN2 the half-hour index after the start of the event
            % P_id_n: duration of each event
            % P_id_a: rainfall amount of each event
            % P_id_max: maximum rainfall amount of each event
            % Pt: half-hourly rainfall time series
            
            P_id(nn)=k;
            FN(nn)=1:length(nn);
            P_id_n(k)=length(nn);
            P_id_a(k)=sum(P(nn));
            P_id_s(k)=submat(hours(nn),1);
            P_id_max(k)=max(P(nn));
            
            Pt=P(nn);
            Wt=WS(nn);
            WSmax(k)=mean(Wt(Pt==max(Pt)));
            %             fpart=fpar(nn);
            %             fpmax(k)=mean(fpar(Pt==max(Pt)));
            k=k+1;
        end
    end
    P_date1    = unique(P_date1); % rain index
    P_date1(P_date1>length(P))=[]; % added by Lian; 有时候往后延12小时会超出总长度
    P_date     = PreNumber;

    P_date(P_date1) = 0;   % no rain index
%     for ii=1:length(P_date1)
%         P_date(P_date==P_date1(ii))=0;
%     end

    if israin
        % with rain
        %     Gs(P_date~=0)      = nan;
        % remove rain events with considerable soil moisture decline
        if exist('SWC')&&any(~isnan(SWC))
            pids=unique(P_id);
            pids=setdiff(pids,0);
            
            for ss=1:length(pids)
                indss=find(P_id==ss);
                
                swcs=SWC(indss);
                ps=P(indss);
                
                if (length(ps)>3)&&any(swcs)
                    [~,pind]=max(ps);
                    pind=pind(end);
                    pind=[pind-1,pind,pind+1];
                    pind(pind<1)=1;
                    pind(pind>length(ps))=length(ps);
                    
                    B=nanmean(swcs(pind));
                    C=nanmean(swcs(end-2:end));

                    delta=(C-B)./B*100;
                    if delta<-10
                        P_id(P_id==ss)=0;
                    end
                end
            end
        end
        
        Gs(P_id==0) = nan; % 有个别点虽然加进来了但是还是没给ID
    else
        % no rain
        Gs(P > 0) = nan; %#ok<UNRCH>
        Gs(P_id~=0)      = nan;
    end

    % added by Xu Lian
    % P_atn: accumulated rainfall over the past n/2 hours
    % P_at2: accumulated rainfall over the past 2 hours
    indxs=[1:6,12,24,48];
    for kk=1:length(indxs)
        eval(['P_at',num2str(indxs(kk)),' = zeros(length(P),1);']);
        P_new=[];
        for n=1:indxs(kk)
            if length(P)-n>=1
                tmp=[nan(n,1);P(1:end-n)];
                P_new=cat(2,P_new,tmp);
            end
        end
        eval(['P_at',num2str(indxs(kk)),'=nansum(P_new,2);']);
    end

    rn=NETRAD*1.59/46/24; % W m-2
    rn(rn<0)=0;
    for kk=1:6
        eval(['W_at',num2str(kk),' = zeros(length(P),1);']);
        W_new=[];
        for n=1:kk
            tmp=[nan(n,1);P(1:end-n)-rn(1:end-n)];
            W_new=cat(2,W_new,tmp);
        end
        eval(['W_at',num2str(kk),'=nansum(W_new,2);']);
        eval(['W_at',num2str(kk),'(W_at',num2str(kk),'<0)=0;']);
    end

    Rs = REMOVE_OUTLIER(Rs);
    Ra = REMOVE_OUTLIER(Ra);
    
%     Gs(isnan(SWC))=nan;
%     Gs(isnan(fpar))=nan;
%     Gs(isnan(Ca))=nan;
    Gs(isnan(Ra))=nan;
    Gs(LEclose<0)=nan; % added by Xu Lian
%     Gs(LEclose>1200)=nan; % added by Xu Lian
%     Gs(isinf(LEclose>1200))=nan; % added by Xu Lian
%     Gs(isnan(G))=nan;
%     Gs(isnan(PPFD_IN))=nan;

    % get the lai time series from MODIS
    siteindex=findstritem(Fluxnet_info.Site_name,sitecode);
    lai=lai_site(:,siteindex)*lairatio;
    lai_full=repmat(lai(1:46),[26,1]);
    s=length(lai);
    while(isnan(lai(s)))
        lai(s)=lai_full(s);
        s=s-1;
    end
    lai_hh = interp1(lai_date,lai,DateNumber,'cubic');
    
    if all(isnan(Ca))
        Ca = interp1(co2date,co2,DateNumber,'cubic');
    end
    
    P_id_org=cat(1,P_id_org,P_id);
%     P_id_a_org=cat(1,P_id_a_org,P_id_a);
%     P_id_n_org=cat(1,P_id_n_org,P_id_n);
    P_org=cat(1,P_org,P);
    CP_org=cat(1,CP_org,CP);
    Ta_org=cat(1,Ta_org,TA);
    WS_org=cat(1,WS_org,WS);
    Rn_org=cat(1,Rn_org,NETRAD);
    if exist('SWC')
        SWC_org=cat(1,SWC_org,SWC);
    else
        SWC_org=cat(1,SWC_org,nan(length(P),1));
    end
    FN_org=cat(1,FN_org,FN);
    LEclose_org=cat(1,LEclose_org,LEclose);
    removeind=cat(1,removeind,isnan(Gs));
    
    % for measurements with G==nan, use H+LE-NETRAD instead--- added by Xu
%     delt=H+LE-NETRAD;
%     G(isnan(G))=delt(isnan(G));

%     fpar(isnan(Gs))=[];
    if exist('SWC')
        SWC(isnan(Gs))=[];
    end
    VPD(isnan(Gs))=[];
    PFT(isnan(Gs))=[];
    TA(isnan(Gs))=[];
    Ca(isnan(Gs))=[];
    NETRAD(isnan(Gs))=[];
    G(isnan(Gs))=[];
    LE(isnan(Gs))=[];
    WS(isnan(Gs))=[];
    PA(isnan(Gs))=[];
    RH(isnan(Gs))=[];
    GPP(isnan(Gs))=[];
    Rs(isnan(Gs))=[];
    Ra(isnan(Gs))=[];
    Delta(isnan(Gs))=[];
    Density(isnan(Gs))=[];
    r(isnan(Gs))=[];
    USTAR(isnan(Gs))=[];
    H(isnan(Gs))=[];
    LEclose(isnan(Gs))=[];
    BR(isnan(Gs))=[];
    Rscorr(isnan(Gs))=[];
    Gscorr(isnan(Gs))=[];
%     FPAR_site(isnan(Gs))=[];
%     PPFD_IN(isnan(Gs))=[];
%     PPFD_OUT(isnan(Gs))=[];
    h_measure(isnan(Gs))=[];
    h_canopy(isnan(Gs))=[];
    CP(isnan(Gs))=[];
    P(isnan(Gs))=[];
    Timescale(isnan(Gs))=[];
%     soil_property_WRB(isnan(Gs))  = [];
%     soil_property_USDA(isnan(Gs)) = [];
    DateNumber(isnan(Gs)) = [];
    P_id(isnan(Gs)) = [];
    SN(isnan(Gs))=[];
    FN(isnan(Gs))=[];
    %Remove outliers for Rs,Ra
    for kk=1:6
        eval(['P_at',num2str(kk),'(isnan(Gs))=[];']);
        eval(['W_at',num2str(kk),'(isnan(Gs))=[];']);
    end
    P_at12(isnan(Gs))=[];
    P_at24(isnan(Gs))=[];
    P_at48(isnan(Gs))=[];
    
    lai_hh(isnan(Gs))=[];
    
    Gs(isnan(Gs))=[];

%     FPAR_ML = [FPAR_ML;fpar];
    VPD_ML  = [VPD_ML;VPD];
    PFT_ML  = [PFT_ML;PFT];
    TA_ML   = [TA_ML;TA];
    if exist('SWC')
        SM_ML   = [SM_ML;SWC];
    else
        SM_ML   = [SM_ML;nan(length(TA),1)];
    end
    if length(SM_ML)~=length(TA_ML)
        break;
    end
    CA_ML   = [CA_ML;Ca];
    RN_ML   = [RN_ML;NETRAD];
    G_ML    = [G_ML; G];
    LE_ML   = [LE_ML;LE];
    WS_ML   = [WS_ML;WS];
    PA_ML   = [PA_ML;PA];
    RH_ML   = [RH_ML;RH];
    GPP_ML  = [GPP_ML;GPP];
    GS_ML   = [GS_ML;Gs];
    Rs_ML   = [Rs_ML;Rs];
    Ra_ML   = [Ra_ML;Ra];
    LAI_ML   = [LAI_ML;lai_hh];
    for kk=1:6
        eval(['P_at',num2str(kk),'_ML   = [P_at',num2str(kk),'_ML;P_at',num2str(kk),'];']); % P_at1_ML   = [P_at1_ML;P_at1];
        eval(['W_at',num2str(kk),'_ML   = [W_at',num2str(kk),'_ML;W_at',num2str(kk),'];']);
    end
    P_at12_ML   =[P_at12_ML;P_at12];
    P_at24_ML   =[P_at24_ML;P_at24];
    P_at48_ML   =[P_at48_ML;P_at48];
    
    Delta_ML   =[Delta_ML;Delta];
    Density_ML =[Density_ML;Density];
    r_ML       =[r_ML;r];
    USTAR_ML      =[USTAR_ML;USTAR];
    H_ML    = [H_ML;H];
    LEclose_ML = [LEclose_ML;LEclose];
    BR_ML   = [BR_ML;BR];
    Rscorr_ML = [Rscorr_ML;Rscorr];
    Gscorr_ML = [Gscorr_ML;Gscorr];
%     FPAR_site_ML = [FPAR_site_ML;FPAR_site];
%     PPFD_IN_ML = [PPFD_IN_ML;PPFD_IN];
%     PPFD_OUT_ML = [PPFD_OUT_ML;PPFD_OUT];
    h_measure_ML = [h_measure_ML;h_measure];
    h_canopy_ML = [h_canopy_ML;h_canopy];
    CP_ML = [CP_ML;CP];
    P_ML=[P_ML; P];
    Pid_ML=[Pid_ML; P_id];
    SN_ML=[SN_ML;SN];
    Timescale_ML=[Timescale_ML;Timescale];
%     soil_property_WRB_ML  = [soil_property_WRB_ML;soil_property_WRB];
%     soil_property_USDA_ML = [soil_property_USDA_ML;soil_property_USDA];
    DateNumber_ML = [DateNumber_ML;DateNumber];    % Date year-month-day
    FN_ML=[FN_ML;FN];
    SiteNumber_ML = cat(1,SiteNumber_ML,ones(length(DateNumber),1)*i);
    date1_ML=cat(1,date1_ML,datestr(DateNumber,'yyyy-mm-dd'));

    clearvars -except data Fluxnet_info FPAR_ML SM_ML VPD_ML PFT_ML TA_ML CA_ML RN_ML ...
        G_ML LE_ML WS_ML PA_ML RH_ML GPP_ML GS_ML CP_ML Rs_ML Ra_ML Delta_ML Density_ML r_ML ...
        USTAR_ML H_ML LEclose_ML BR_ML Rscorr_ML Gscorr_ML h_measure_ML h_canopy_ML PPFD_IN_ML ...
        soil_property_WRB_ML soil_property_USDA_ML DateNumber_ML SiteNumber_ML P_ML SN_ML FN_ML Timescale_ML Pid_ML P_id_n P_id_a P_id_max2 k sn ...
        P_at1_ML P_at2_ML P_at3_ML P_at4_ML P_at5_ML P_at6_ML P_at12_ML P_at24_ML P_at48_ML ...
        W_at1_ML W_at2_ML W_at3_ML W_at4_ML W_at5_ML W_at6_ML W_at12_ML ...
        date1_ML lai_site lai_date LAI_ML israin co2date co2 ...
        P_id_org CP_org P_org Rn_org SWC_org Ta_org WS_org LEclose_org removeind FN_org P_id_s
%         FPAR_site_ML PPFD_IN_ML PPFD_OUT_ML ...
end
for i=1:length(P_id_n)
    Pidn(i)=length(Pid_ML(Pid_ML==i));
    if Pidn(i)>0
        Pida(i)=sum(P_ML(Pid_ML==i));
    else  Pida(i)=0;
    end
end
PPn=Pidn./P_id_n;
PPa=Pida./P_id_a;
VPn=sum(PPn>0.8 & PPa>0.8);
fprintf('The available P events are %d \n',VPn)

% delete those with imcomplete data

vars={'VPD','PFT','TA','RN','CA','G','LE','WS',...
    'PA','RH','GPP','GS','Rs','Ra','LAI','CP','LEclose','P',...
    'P_at1','P_at2','P_at3','P_at4','P_at5','P_at6',...
    'W_at1','W_at2','W_at3','W_at4','W_at5','W_at6',...
    'Delta','Density','r','H','USTAR','BR',...
    'Rscorr','Gscorr','h_measure','h_canopy','Pid',...
    'SN','Timescale','DateNumber','FN','SiteNumber',...
    'date1','SM'}; % FPAR, PPFD_IN, PPFD_OUT, soil_property_WRB, soil_property_USDA

if  israin
     
    save rain_events_hourly.mat P_id_org CP_org LEclose_org P_org Rn_org SWC_org Ta_org WS_org FN_org -append
else
    % remove the sites with rainfall over the past 48 hours (just small rainfall)
    isremove=((P_ML>0)|(P_at1_ML>0)|(P_at2_ML>0)|(P_at3_ML>0)|(P_at4_ML>0)|(P_at5_ML>0)|(P_at6_ML>0)); %#ok<UNRCH>
    isremove2=((P_ML>0)|(P_at48_ML>0)); 
    for kk=1:length(vars)
        eval([vars{kk},'_ML(find(isremove2),:)=[];']);
    end
end


% save the data to netcdf
if israin
    file_input = './ncfile/Inputva_withrain_hourly.nc'; % _RHcorr
else
    file_input = './ncfile/Inputva_norain_hourly.nc';
end

nccreate(file_input,'LAI','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'LAI', LAI_ML);
nn(1)=length(PFT_ML(isnan(LAI_ML)));

nccreate(file_input,'SM','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'SM', SM_ML);
nn(2)=length(PFT_ML(isnan(SM_ML)));

nccreate(file_input,'VPD','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'VPD', VPD_ML);
nn(3)=length(PFT_ML(isnan(VPD_ML)));
 
nccreate(file_input,'PFT','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'PFT', PFT_ML);
nn(4)=length(PFT_ML(isnan(PFT_ML)));
 
nccreate(file_input,'TA','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'TA', TA_ML);
nn(5)=length(PFT_ML(isnan(TA_ML)));
 
nccreate(file_input,'Ca','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'Ca', CA_ML);
nn(6)=length(PFT_ML(isnan(CA_ML)));
 
nccreate(file_input,'Rn','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'Rn', RN_ML);
nn(7)=length(PFT_ML(isnan(RN_ML)));
 
nccreate(file_input,'G','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'G', G_ML);
nn(8)=length(PFT_ML(isnan(G_ML)));

nccreate(file_input,'LE','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'LE', LE_ML);
 
nccreate(file_input,'WS','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'WS', WS_ML);
nn(9)=length(PFT_ML(isnan(WS_ML)));
 
nccreate(file_input,'PA','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'PA', PA_ML);
nn(10)=length(PFT_ML(isnan(PA_ML)));
 
nccreate(file_input,'RH','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'RH', RH_ML);
nn(11)=length(PFT_ML(isnan(RH_ML)));

nccreate(file_input,'GPP','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'GPP', GPP_ML);

nccreate(file_input,'Gs','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'Gs', GS_ML);
%
nccreate(file_input,'Rs','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'Rs', Rs_ML);

nccreate(file_input,'Ra','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'Ra', Ra_ML);
nn(12)=length(PFT_ML(isnan(Ra_ML)));
%
nccreate(file_input,'Delta','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'Delta', Delta_ML);
nn(13)=length(PFT_ML(isnan(Delta_ML)));
%
nccreate(file_input,'Density','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'Density', Density_ML);
nn(14)=length(PFT_ML(isnan(Density_ML)));
%
nccreate(file_input,'r','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'r', r_ML);
nn(15)=length(PFT_ML(isnan(r_ML)));

% nccreate(file_input,'USTAR','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
% ncwrite(file_input,'USTAR', USTAR_ML);
%
nccreate(file_input,'H','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'H', H_ML);

nccreate(file_input,'LEclose','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'LEclose', LEclose_ML);
nn(16)=length(PFT_ML(isnan(LEclose_ML)));
%
% nccreate(file_input,'BR','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
% ncwrite(file_input,'BR', BR_ML);
%
nccreate(file_input,'Rscorr','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'Rscorr', Rscorr_ML);

nccreate(file_input,'Gscorr','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'Gscorr', Gscorr_ML);

%
nccreate(file_input,'h_canopy','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'h_canopy', h_canopy_ML);
nn(18)=length(PFT_ML(isnan(h_canopy_ML)));

nccreate(file_input,'CP','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'CP', CP_ML);
nn(18)=length(PFT_ML(isnan(CP_ML)));
 
nccreate(file_input,'Timescale','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'Timescale', Timescale_ML);
nn(20)=length(PFT_ML(isnan(Timescale_ML)));
 
nccreate(file_input,'SN','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'SN', SN_ML);
nn(22)=length(PFT_ML(isnan(SN_ML)));
 
if israin
    nccreate(file_input,'P','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
    ncwrite(file_input,'P', P_ML);
    nn(19)=length(PFT_ML(isnan(P_ML)));
    
    nccreate(file_input,'P_id','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
    ncwrite(file_input,'P_id', Pid_ML);
    % nn(21)=length(PFT_ML(isnan(Pid_ML)));
    
    nccreate(file_input,'P_id_n','Dimensions', { 'num2', length(P_id_n)},'Datatype','double');
    ncwrite(file_input,'P_id_n', P_id_n);
    
    nccreate(file_input,'P_id_s','Dimensions', { 'num2', length(P_id_n)},'Datatype','double');
    ncwrite(file_input,'P_id_s', P_id_s);
    
    nccreate(file_input,'P_id_a','Dimensions', { 'num2', length(P_id_a)},'Datatype','double');
    ncwrite(file_input,'P_id_a', P_id_a);
    
    nccreate(file_input,'P_at1','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
    ncwrite(file_input,'P_at1', P_at1_ML);
    nccreate(file_input,'P_at2','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
    ncwrite(file_input,'P_at2', P_at2_ML);
    nccreate(file_input,'P_at3','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
    ncwrite(file_input,'P_at3', P_at3_ML);
    nccreate(file_input,'P_at4','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
    ncwrite(file_input,'P_at4', P_at4_ML);
    nccreate(file_input,'P_at5','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
    ncwrite(file_input,'P_at5', P_at5_ML);
    nccreate(file_input,'P_at6','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
    ncwrite(file_input,'P_at6', P_at6_ML);
    
    nccreate(file_input,'W_at1','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
    ncwrite(file_input,'W_at1', W_at1_ML);
    nccreate(file_input,'W_at2','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
    ncwrite(file_input,'W_at2', W_at2_ML);
    nccreate(file_input,'W_at3','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
    ncwrite(file_input,'W_at3', W_at3_ML);
    nccreate(file_input,'W_at4','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
    ncwrite(file_input,'W_at4', W_at4_ML);
    nccreate(file_input,'W_at5','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
    ncwrite(file_input,'W_at5', W_at5_ML);
    nccreate(file_input,'W_at6','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
    ncwrite(file_input,'W_at6', W_at6_ML);
end
% nccreate(file_input,'soil_property_WRB','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
% ncwrite(file_input,'soil_property_WRB', soil_property_WRB_ML);

% nccreate(file_input,'soil_property_USDA','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
% ncwrite(file_input,'soil_property_USDA', soil_property_USDA_ML);

nccreate(file_input,'DateNumber','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'DateNumber', DateNumber_ML);
nn(23)=length(PFT_ML(isnan(DateNumber_ML)));

nccreate(file_input,'SiteNumber','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'SiteNumber', SiteNumber_ML);
nn(24)=length(PFT_ML(isnan(DateNumber_ML)));

nccreate(file_input,'FN','Dimensions', { 'num', length(PFT_ML)},'Datatype','double');
ncwrite(file_input,'FN', FN_ML);
% nn(24)=length(PFT_ML(isnan(FN_ML)));

nccreate(file_input,'date1','Dimensions', { 'num', length(PFT_ML), 'c', 10},'Datatype','char');
ncwrite(file_input,'date1', date1_ML);
