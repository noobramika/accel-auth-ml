
Acc_Data_All_Users = cell(1, 10);

% Load and store data for each user
for nc = 1:10
    % Load each user's data file, and extract the required features
    T_Acc_Data_Day1 = load(sprintf('U%02d_Acc_FreqD_FDay.mat', nc));
    
    Temp_Acc_Data = T_Acc_Data_Day1.Acc_FD_Feat_Vec(1:36, 1:43);
    
    % array
    Acc_Data_All_Users{nc} = Temp_Acc_Data;
end

% Calculate variance for each feature across all users
inter_user_variance = zeros(36, 43);

for row = 1:36
    for col = 1:43
        % Extract feature values for all users for the given row and column
        feature_values = zeros(1, 10);
        
        for nc = 1:10
            feature_values(nc) = Acc_Data_All_Users{nc}(row, col);
        end
        
        % this will calculate variance for the current feature across users
        inter_user_variance(row, col) = var(feature_values);
    end
end

% this will calculate the average variance
average_variance_per_feature = mean(inter_user_variance, 1);

% this will plot the average inter user variance per feature
figure;
plot(1:43, average_variance_per_feature, '-o');
xlabel('Feature Index');
ylabel('Average Inter-user Variance');
title('Average Inter-user Variance for FreqD_FDay');
grid on;