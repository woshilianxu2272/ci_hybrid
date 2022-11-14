
%% Read the time series data from each fluxnet (close path)
ccc
Fluxnet_info = readtable('.\FLUXNET2015\FULLSET\Fluxnet_sites_Hysteresis.csv');

for i=1:size(Fluxnet_info,1)
    
    list_files  = dir(['.\FLUXNET2015\FULLSET\fluxnet2015\*',Fluxnet_info.Site_name{i},'*']);  % "FULLSET" is the folder containing the fluxnet data in .zip format
    %     list_files=list_files(3:end);
    
    if ~isempty(list_files)
        zipFilename = list_files(1).name;
    else
        continue;
    end
    
    strs=regexp(zipFilename,'\_','split');
    sitename=strs{2};
    
%     if exist(['MAT_DATA_hourly/',sitename,'.mat'])
%         continue;
%     end

    file=dir(['.\FLUXNET2015\*',sitename,'*']);

    % read csv and xls files
    filename_csv = dir([file(1).folder,'\',file(1).name,'\*.csv']);
    PREFIX   =    filename_csv(1).name(1:3);
    if(strcmp(PREFIX,'FLX')) % Fluxnet 2015 dataset
        title_name = filename_csv(1).name(1:end-4);
        for ii=1:length(title_name) 
           if(strcmp(title_name(ii),'_'))
               title_name(ii) = ' ';
           end
        end
        sitecode = title_name(5:10);
    end
    clear ii;
    
    
    % read canopy height data and other info
    ind = findstritem(Fluxnet_info.Site_name,sitecode);
    if ~isempty(ind)
        cover_type   = Fluxnet_info.Cover_type{ind};
        h.canopy     = Fluxnet_info.Canopy_hnew(ind);  % 82���ϵĵ�û���ṩ������������������Щָ����Gs���㲻��
        h.measure    = Fluxnet_info.Tower_hnew(ind);
        latitude     = Fluxnet_info.Latitude(ind);
        longitude    = Fluxnet_info.Longitude(ind);
        elevation    = Fluxnet_info.Elevation(ind);
        
        % exlucde sites where the LUCC doesn't match satellite-observed values 
        mod_lucc    = Fluxnet_info.MOD_LUCC(ind);
        mod_lucc=mod_lucc{1};
        if strcmp(mod_lucc,'WSA')
            mod_lucc='SAV';
        elseif strcmp(mod_lucc,'OSH')||strcmp(mod_lucc,'CSH')
            mod_lucc='SHU';
        end
        if strcmp(cover_type,'WSA')
            cover_type='SAV';
        elseif strcmp(cover_type,'OSH')||strcmp(cover_type,'CSH')
            cover_type='SHU';
        end
%         if strcmp(mod_lucc,'SHU')&&strcmp(cover_type,'SHU')
%             break;
%         else
%             continue;
%         end
        if ~strcmp(mod_lucc, cover_type)
            disp(['Site ',sitecode,' doesnt match satellite-observed LUCC']);
            continue;
        end
        if isnan(h.canopy)||isnan(h.measure)
            continue;
        end
    else
        continue;
    end
    clear ind;
    
    % read the hourly data
    filename_csv = dir(fullfile(file(1).folder,file(1).name,'\*FULLSET_HH*.csv')); % use half hourly data
    file_flag    = 0;
    if(isempty(filename_csv))
        filename_csv = dir(fullfile(file(1).folder,file(1).name,'*FULLSET_HR*.csv'));
        file_flag    = 1; % use hourly data
    end

    csv_file = readtable([fullfile(file(1).folder,file(1).name, filename_csv(1).name)]);
    
    H              = csv_file.H_CORR;          % Sensible heat flux, W/m2
    if all(H==-9999)
        H          = csv_file.H_F_MDS;
    end
    H_quality      = csv_file.H_F_MDS_QC;      % 0 = measured; 1 = good quality gapfill; 2 = medium; 3 = poor
    LE             = csv_file.LE_CORR;         % latent heat flux, W/m2
    if all(LE==-9999)
        LE2         = csv_file.LE_F_MDS;
    end
    LE_quality     = csv_file.LE_F_MDS_QC;     % 0 = measured; 1 = good quality gapfill; 2 = medium; 3 = poor
    G              = csv_file.G_F_MDS;         % soil heat flux, W/m2
    G_quality      = csv_file.G_F_MDS_QC;      % 0 = measured; 1 = good quality gapfill; 2 = medium; 3 = poor
    TA             = csv_file.TA_F;            % air temperature, deg C
    TA_quality     = csv_file.TA_F_QC;         % 0 = measured; 1 = good quality gapfill; 2 = downscaled from ERA
    SW_IN          = csv_file.SW_IN_F;         % shortwave radiation incoming, W/m2
    SW_IN_quality  = csv_file.SW_IN_F_QC;      % 0 = measured; 1 = good quality gapfill; 2 = downscaled from ERA
    VPD            = csv_file.VPD_F;           % vapor pressure decifit, hPa
    VPD_quality    = csv_file.VPD_F_QC;        % 0 = measured; 1 = good quality gapfill; 2 = downscaled from ERA
    PA             = csv_file.PA_F;            % atmospheric pressure, kPa
    PA_quality     = csv_file.PA_F_QC;         % 0 = measured; 2 = downscaled from ERA
    P              = csv_file.P_F;             % precipitation, mm
    P_quality      = csv_file.P_F_QC;          % 0 = measured; 2 = downscaled from ERA
    WS             = csv_file.WS_F;            % wind speed, m/s
    WS_quality     = csv_file.WS_F_QC;         % 0 = measured; 2 = downscaled from ERA
    if ismember('RH',csv_file.Properties.VariableNames)
        RH             = csv_file.RH;              % relative humidity
    else
        RH             = 100*(1-VPD./(0.6108*exp((17.27*TA)./(TA+237.3))));
    end
    if ismember('NETRAD',csv_file.Properties.VariableNames)
        % necessary variable for caluclating LE_close
        NETRAD         = csv_file.NETRAD;          % net radiation, Rn, W/m2
    else
        clearvars -except list_files i Fluxnet_info
        continue;
    end
    USTAR          = csv_file.USTAR;           % friction velocity, m/s
    if ismember('PPFD_OUT',csv_file.Properties.VariableNames)
        flag_PAR_out = 1;
        PPFD_OUT          = csv_file.PPFD_OUT;         % photosynthetic photon flux density outcoming, W/m2 (PAR) 
    else
        flag_PAR_out = 0;
    end
    if ismember('PPFD_IN',csv_file.Properties.VariableNames)
        flag_PAR = 1;
        PPFD_IN           = csv_file.PPFD_IN;         % photosynthetic photon flux density incoming, W/m2 (PAR)
    else
        flag_PAR = 0;
    end
    GPP            = csv_file.GPP_NT_VUT_REF;   % GPP, umol CO2/m2/s
    GPP_quality    = csv_file.NEE_VUT_REF_QC;   % 0 = measured; 1 = good quality gapfill; 2 = medium; 3 = poor
    flag_SWC_1     = 0; 
    flag_SWC_2     = 0;
    if ismember('SWC_F_MDS_1',csv_file.Properties.VariableNames)
        flag_SWC_1     = 1;
        SWC_1          = csv_file.SWC_F_MDS_1;     % soil water content (volumetric), (SWC layer 1 - upper), %
        SWC_1_quality  = csv_file.SWC_F_MDS_1_QC;  % 0 = measured; 1 = good quality gapfill; 2 = medium; 3 = poor
    end
    if ismember('SWC_F_MDS_2',csv_file.Properties.VariableNames)
        flag_SWC_2     = 1;
        SWC_2          = csv_file.SWC_F_MDS_2;     % soil water content (volumetric), (SWC layer 2 - lower), %
        SWC_2_quality  = csv_file.SWC_F_MDS_2_QC;  % 0 = measured; 1 = good quality gapfill; 2 = medium; 3 = poor
    end
    flag_SWC = 0;
    % use upper soil moisture for priority
    if flag_SWC_1==1
        if ~all(SWC_1==-9999)
            SWC = SWC_1;
            SWC_quality = SWC_1_quality;
            flag_SWC = 1;
        end
    elseif flag_SWC_2==1
        if ~all(SWC_2==-9999)
            SWC = SWC_2;
            SWC_quality = SWC_2_quality;
            flag_SWC = 2;
        end
    end
    clear flag_SWC_1; clear flag_SWC_2;
    clear SWC_1; clear SWC_2;
    if ismember('CO2_F_MDS',csv_file.Properties.VariableNames)
        Ca         = csv_file.CO2_F_MDS;   % CO2 concentration  umol/mol
        Ca_quality = csv_file.CO2_F_MDS_QC;
        flag_Ca    = 1;
    else
        flag_Ca    = 0;
    end
    time           = num2str(csv_file.TIMESTAMP_START);
    
    if ismember('LW_OUT',csv_file.Properties.VariableNames)
        flag_LW_OUT = 1;
    else
        flag_LW_OUT = 0;
    end
    if flag_LW_OUT==1
        LW_OUT           = csv_file.LW_OUT;         % longwave radiation outgoing, W/m2 (PAR)
    end
    SW_IN_POT          = csv_file.SW_IN_POT;         % shortwave radiation incoming, W/m2
    
    year     = str2num(time(:,1:4));
    month    = str2num(time(:,5:6));
    day      = str2num(time(:,7:8));
    hours    = str2num(time(:,9:10));
    minutes  = str2num(time(:,11:12));
    
%% Step 1: Quality control -- remove data in poor quality
    H(H<-9000)              = NaN;
    %     H(H_quality>=2)         = NaN;  % remove data in poor quality
    LE(LE<-9000)            = NaN;
    %     LE(LE_quality>=2)       = NaN;  % remove data in poor quality
    G(G<-9000)              = NaN;
    %     G(G_quality>=2)         = NaN;  % remove data in poor quality
    SW_IN(SW_IN<-9000)      = NaN;
    %     SW_IN(SW_IN_quality==2) = NaN;  % remove data in poor quality
    TA(TA<-9000)            = NaN;
    %     TA(TA_quality==2)       = NaN;  % remove data in poor quality
    WS(WS<-9000)            = NaN;
    %     WS(WS_quality==2)       = NaN;  % remove data in poor quality
    VPD(VPD<-9000)          = NaN;
    %     if ~all(VPD_quality==2)
    %     VPD(VPD_quality==2)     = NaN;  % remove data in poor quality
    %     end
    PA(PA<-9000)            = NaN;
    %     if ~all(PA_quality==2)
    %     PA(PA_quality==2)       = NaN;  % remove data in poor quality
    %     end
    P(P<-9000)              = NaN;
    %     P(P_quality==2)         = NaN;  % remove data in poor quality
    RH(RH<-9000)            = NaN;
    NETRAD(NETRAD<-9000)    = NaN;
    USTAR(USTAR<-9000)      = NaN;
    if flag_PAR==1
        PPFD_IN(PPFD_IN<-9000)        = NaN;
    end
    if flag_PAR_out==1
        PPFD_OUT(PPFD_OUT<-9000)        = NaN;
    end
    GPP(GPP<-1000)          = NaN;
    %     GPP(GPP_quality>=2)     = NaN;  % remove data in poor quality
    if flag_SWC~=0
        SWC(SWC<-9000)      = NaN;
        %         SWC(SWC_quality>=2) = NaN;
    end

    if flag_Ca==1
        Ca(Ca<-9000)      = NaN;
    end
    
    if flag_LW_OUT==1
        LW_OUT(LW_OUT<-9000)        = NaN;
    end
    SW_IN_POT(SW_IN_POT<-9000)        = NaN;
    
%% step 2: from half hourly to houly
    if file_flag==0   % half hour
        H = reshape(H, 2, []);
        H = transpose(nanmean(H, 1));
        
        LE = reshape(LE, 2, []);
        LE = transpose(nanmean(LE, 1));
        
        G = reshape(G, 2, []);
        G = transpose(nanmean(G, 1));
        
        SW_IN = reshape(SW_IN, 2, []);
        SW_IN = transpose(nanmean(SW_IN, 1));
        
        TA = reshape(TA, 2, []);
        TA = transpose(nanmean(TA, 1));
        
        WS = reshape(WS, 2, []);
        WS = transpose(nanmean(WS, 1));
        
        VPD = reshape(VPD, 2, []);
        VPD = transpose(nanmean(VPD, 1));
        
        PA = reshape(PA, 2, []);
        PA = transpose(nanmean(PA, 1));
        
        P = reshape(P, 2, []); % accumulated precipitation
        %         P = transpose(nansum(P, 1));
        P = transpose(sum(P, 1));
        
        RH = reshape(RH, 2, []);
        RH = transpose(nanmean(RH, 1));
        
        NETRAD = reshape(NETRAD, 2, []);
        NETRAD = transpose(nanmean(NETRAD, 1));
        
        USTAR = reshape(USTAR, 2, []);
        USTAR = transpose(nanmean(USTAR, 1));
        
        if flag_PAR==1
            PPFD_IN = reshape(PPFD_IN, 2, []);
            PPFD_IN = transpose(nanmean(PPFD_IN, 1));
        end
        if flag_PAR_out==1
            PPFD_OUT = reshape(PPFD_OUT, 2, []);
            PPFD_OUT = transpose(nanmean(PPFD_OUT, 1));
        end
        GPP = reshape(GPP, 2, []);
        GPP = transpose(nanmean(GPP, 1));
        %     GPP(GPP_quality>=2)     = NaN;  % remove data in poor quality
        if flag_SWC~=0
            SWC = reshape(SWC, 2, []);
            SWC = transpose(nanmean(SWC, 1));
        end
    
        if flag_Ca==1
            Ca = reshape(Ca, 2, []);
            Ca = transpose(nanmean(Ca, 1));
        end
        
        if flag_LW_OUT==1
            LW_OUT = reshape(LW_OUT, 2, []);
            LW_OUT = transpose(nanmean(LW_OUT, 1));
        end
        SW_IN_POT = reshape(SW_IN_POT, 2, []);
        SW_IN_POT = transpose(nanmean(SW_IN_POT, 1));
    
        hours = reshape(hours, 2, []);
        hours = transpose(nanmean(hours, 1));
        
        month = reshape(month, 2, []);
        month = transpose(nanmean(month, 1));
        
        year = reshape(year, 2, []);
        year = transpose(nanmean(year, 1));
        
        day = reshape(day, 2, []);
        day = transpose(nanmean(day, 1));
    end
    file_flag = 1;
    
    % create date info only records the year, month, day
    if length(unique(day))>1
        DateNumber   = datenum(year,month,day);
    else DateNumber  = datenum(year,month,ones(length(year),1));
        for j=1:length(DateNumber)/24
            DateNumber((j-1)*24+1:j*24)=DateNumber(1)+j-1;
        end
    end
    PreNumber   =1:1:length(day);PreNumber=PreNumber';
    year_number  = unique(year);
    
%% Step 3: choose daytime (shortwave>50, sensible heat>5)
    GPP(isnan(H))      = NaN;
    GPP(isnan(SW_IN))  = NaN;
    VPD(isnan(H))      = NaN;
    VPD(isnan(SW_IN))  = NaN;
    LE(isnan(H))       = NaN;
    LE(isnan(SW_IN))   = NaN;
    
%     GPP(SW_IN<=50)            = NaN; % filtering daytime
%     GPP(H<=5)                 = NaN;
%     VPD(SW_IN<=50)            = NaN;
%     VPD(H<=5)                 = NaN;
%     LE(SW_IN<=50)             = NaN;
%     LE(H<=5)                  = NaN;
% %     P(SW_IN<=50)              = NaN;
% %     P(H<=5)                    = NaN;
    
    GPP(GPP<=0)               = NaN;
    LE(LE<=0)                 = NaN;
%     P(LE<=0)                  = NaN;
    VPD                       = VPD/10; % hPD -> kPa
    VPD(VPD<=0.01)               = NaN;    % only exclude very low VPD
%     P(VPD<=0.01)               = NaN;

%% Step 4: Choose growing season, when 15-day moving average GPP was higher than 
%     %         half of the 95th percentile of the daily GPP within the site year
%     GPP_original  = GPP;
%     Date       = unique(DateNumber);
%     Date_used  = zeros(length(Date),1);
%     temp_GPP   = zeros(length(Date),1)*NaN;
%     for d=1:length(Date)
%         temp_GPP(d)  = nanmean(GPP_original(DateNumber==Date(d)));
%     end
%     for y=1:length(year_number)
%         Date_year = unique(DateNumber(year==year_number(y)));
%         START     = find(Date==Date_year(1));
%         END       = find(Date==Date_year(end));
%         if length(find(~isnan(temp_GPP(START:END))))>15
%             temp_2 = 0.5*quantile(temp_GPP(START:END),0.95);
%             for d=1:length(Date_year)
%                 if (y==1 && d<8) || (y==length(year_number) && d>length(Date_year)-7)
%                     continue;
%                 end
%                 t = find(Date==Date_year(d));
%                 g = nanmean(temp_GPP(t-7:t+7));
%                 if g>=temp_2
%                     Date_used(t) = Date(t);
%                 end
%             end
%         end
%     end
%     Date_used(Date_used==0) =[]; % days used
%     Date_unused             = setdiff(Date,Date_used); % days not used
%     for d=1:length(Date_unused)
%         LE(DateNumber==Date_unused(d))    = NaN;
%         VPD(DateNumber==Date_unused(d))   = NaN;
%         GPP(DateNumber==Date_unused(d))   = NaN;
%         Ca(DateNumber==Date_unused(d))    = NaN;
%         TA(DateNumber==Date_unused(d))    = NaN;
%         SWC(DateNumber==Date_unused(d))   = NaN;
%         P(DateNumber==Date_unused(d))     = 0;
%         if flag_PAR==1 && flag_PAR_out==1
%             PPFD_IN(DateNumber==Date_unused(d))  = NaN;
%             PPFD_OUT(DateNumber==Date_unused(d))  = NaN;
%         end
%     end
%     clear Date; clear d; clear y; clear temp_GPP; clear Date_year;
%     clear START; clear END; clear temp_2; clear t; clear g;

%% step 5: choose days higher than 0oC degree
    GPP(TA<=0)            = NaN; % filtering
    VPD(TA<=0)            = NaN;
    LE(TA<=0)             = NaN;
%     P(TA<=2)              = 0;
    if flag_SWC~=0
        SWC(SWC<=0)           = NaN;
%         P(SWC<=0)             = 0;
    end

%% Step 2: Exclude rainy days (and the following 48h) regarding GEP,VPD,PPFD,LE or so
%     P_date    = unique(PreNumber(P>0));
%     P_date1   = P_date;
%     if file_flag    == 0
%         for ni=1:2*nn
%             P_date1    = [P_date1;P_date+ni];
%             P_date1    = unique(P_date1);
% %             P_ind(P_date+ni)
%         end
%     elseif file_flag    == 1
%         for ni=1:nn
%             P_date1    = [P_date1;P_date+ni];
%             P_date1    = unique(P_date1);
%         end
%     end
%     P_date1    = unique(P_date1);   % index of all rain hours
%     P_date     = PreNumber;  % index of all no-rain hours
%     for ii=1:length(P_date1)
%         P_date(P_date==P_date1(ii))=[];
%     end
    
%     % rainfall days excluded in all variables
%     isremove=false(length(PreNumber),1);
%     isremove(P_date)=1;
%     H(isremove)      = NaN;
%     LE(isremove)     = NaN;
%     G(isremove)      = NaN;
%     SW_IN(isremove)  = NaN;
%     TA(isremove)     = NaN;
%     WS(isremove)     = NaN;
%     VPD(isremove)    = NaN;
%     PA(isremove)     = NaN;
%     RH(isremove)     = NaN;
%     NETRAD(isremove) = NaN;
%     USTAR(isremove)  = NaN;
%     GPP(isremove)    = NaN;
%     SWC(isremove)    = NaN;
%     if flag_PAR==1
%         PPFD_IN(isremove)   = NaN;
%     end
%     if flag_PAR_out==1
%         PPFD_OUT(isremove)   = NaN;
%     end
%     if flag_Ca==1
%         Ca(isremove)   = NaN;
%     end
%     H(isremove)      = NaN;

%     for p=1:length(P_date)
%         H(PreNumber==P_date(p))      = NaN;
%         LE(PreNumber==P_date(p))     = NaN;
%         G(PreNumber==P_date(p))      = NaN;
%         SW_IN(PreNumber==P_date(p))  = NaN;
%         TA(PreNumber==P_date(p))     = NaN;
%         WS(PreNumber==P_date(p))     = NaN;
%         VPD(PreNumber==P_date(p))    = NaN;
%         PA(PreNumber==P_date(p))     = NaN;
%         RH(PreNumber==P_date(p))     = NaN;
%         NETRAD(PreNumber==P_date(p)) = NaN;
%         USTAR(PreNumber==P_date(p))  = NaN;
%         GPP(PreNumber==P_date(p))    = NaN;
%         SWC(PreNumber==P_date(p))    = NaN;
%         if flag_PAR==1
%             PPFD_IN(PreNumber==P_date(p))   = NaN;
%         end
%         if flag_PAR_out==1
%             PPFD_OUT(PreNumber==P_date(p))   = NaN;
%         end
%         if flag_Ca==1
%             Ca(PreNumber==P_date(p))   = NaN;
%         end
%         H(PreNumber==P_date(p))      = NaN;
% %         P(PreNumber==P_date(p))      = NaN;
%     end
    clear p;

%% Calculate stomatal resistance

    RH           = RH/100;
    Es           = 0.6108*exp((17.27*TA)./(TA+237.3));        % saturated water vapor pressure, kPa
    Delta        = 4098*Es./((237.3+TA).^2);                  % slope of vapor pressure curve, kPa/C
    Cp           = 1012;                                      % specific heat capacity, J/kg/C
    r            = 10^(-6)*Cp*PA./(0.622*(2.501-0.00236*TA)); % psychrometric constant, kPa/C
    Density      = (PA./(0.28705*(TA+273.15))).*(1-(0.378*Es.*RH./PA));              % density of air, kg/m3
    L            = -(USTAR.^3.*Cp.*Density.*(TA+273.15))./(0.41*9.8*H);     % Monin-Obukhov length, m/s
    ksi          = (h.measure-h.canopy*2/3)./L;                             % stability parameter, dimensionless
    Glob.STAB    = 1;                                         % 1 or 0, to use which correction form
    Glob.K       = 0.41;                                      % von karman constant
    Glob.d       = h.canopy*(2/3);                              % zero plane displacement
%     Glob.z0      = h.canopy*0.1;                              % roughness length
    Glob.z0h      = h.canopy*0.1;                              % roughness length for heat
    Glob.z0m      = Glob.z0h * exp(2);
    ra_min       = aero_resistance(h.measure,h.canopy,WS);    % aerodynamic resistance, s/m
    Ra           = zeros(length(TA),1);
    for n=1:length(Ra)
        Glob.ra_min  = ra_min(n);
        Ra(n)        = U_STAR(h.measure, WS(n), L(n), Glob );       % aerodynamic resistance, s/m
    end
    if all(isnan(G))
        Rs           = ((((Delta.*NETRAD+(1012*Density.*VPD)./Ra)./LE-Delta)./r)-1).*Ra;
        G_flag       = 1;  % whether G is neglected or not. 1-neglect; 0-not neglect;
    else
        Rs           = ((((Delta.*(NETRAD-G)+(1012*Density.*VPD)./Ra)./LE-Delta)./r)-1).*Ra;
        G_flag       = 0;  % whether G is neglected or not. 1-neglect; 0-not neglect;
    end
    clear n;
    
    
    Gs = 1000*1./Rs;  % unit: mm/s
    % VPD_leaf !
    VPD_leaf              = (1000./Gs).*(LE.*r)./(Density*Cp);  % unit: kPa
    VPD_leaf(VPD_leaf<0.01)  = NaN;
    VPD_leaf(VPD_leaf>2*max(VPD)) = NaN;
    Gs = Gs.*PA./(8.314*(273+TA));  % unit transformation: mm/s --> mol/m2/s

    % processing outliers of Gs
    Gs_temp = Gs;
    Gs_temp(isnan(Gs_temp)) = [];
    nannan_Gs = find(~isnan(Gs));
    [y,j,xmedian,xsigma]=hampel(Gs_temp,15,3);
    y(j) = nan;
    Gs(~isnan(Gs)) = y;
    clear y j xmedian xsigma Gs_temp 
   % ????????????????
    Gs(Gs<=0)               = NaN;
       
    clear csv_file;
    clear filename_csv;
    clear title_name;
    clear H_quality;
    clear LE_quality;
    clear G_quality;
    clear TA_quality;
    clear SW_IN_quality;
    clear VPD_quality;
    clear PA_quality;
    clear P_quality;
    clear WS_quality;
    clear GEP_quality;
    clear SWC_1;
    clear SWC_1_quality;
    clear SWC_2;
    clear SWC_2_quality;
    clear Ca_quality;
    
    %        clear Date_used;
    clear Glob;
    clear ksi;
    clear P_date;
    clear PREFIX;
    clear zipFilename;
    clear ra_min;
    clear minutes;

       %% save files
       save(['MAT_DATA_hourly/',sitecode,'.mat'],'-regexp','[^list_files^Fluxnet_info^i^tempdir]');
       
       clear PPFD PPFD_OUT  % ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????
       clearvars -except list_files Fluxnet_info i tempdir nn
end

