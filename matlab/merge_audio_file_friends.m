%%% generate audio feature mat (normalized)

close all;clc;clear;

% toggles off print info and waitbar
mirverbose(0);
mirwaitbar(0);

folder_base = '/home/wyj/dataset/Friends/Friends.%s/speaking-audio';    % big bang theory
foder_to_save = '/home/wyj/dataset/Friends-Audio';
names = {'chandler', 'joey', 'monica', 'phoebe', 'rachel', 'ross'};
videos = {'S01E03', 'S04E04', 'S07E07','S10E15'};

% merge data here
audio_data_all_0 = [];
audio_data_all_1 = [];
audio_data_all_2 = [];
audio_data_all_3 = [];
audio_data_all_4 = [];
audio_data_all_5 = [];
for i=1:5
    for j=1:length(videos)
        audioFile = sprintf('%s/%s.wav', sprintf(folder_base, videos{j}), names{i});
        
        if ~exist(audioFile, 'file')
            continue;
        end
        audioData = miraudio(audioFile);
        
        if i==1
            audio_data_all_0 = [audio_data_all_0; mirgetdata(audioData)];
        elseif i==2
            audio_data_all_1 = [audio_data_all_1; mirgetdata(audioData)];
        elseif i==3
            audio_data_all_2 = [audio_data_all_2; mirgetdata(audioData)];
        elseif i==4
            audio_data_all_3 = [audio_data_all_3; mirgetdata(audioData)];
        elseif i==5
            audio_data_all_4 = [audio_data_all_4; mirgetdata(audioData)];
		elseif i==6
            audio_data_all_5 = [audio_data_all_5; mirgetdata(audioData)];

        end
    end
end

% write feature matrix to disk
sprintf('%s/%s_merged_all.wav', foder_to_save, names{1})
audiowrite(sprintf('%s/%s_merged_all.wav', foder_to_save, names{1}), audio_data_all_0, 16000);
audiowrite(sprintf('%s/%s_merged_all.wav', foder_to_save, names{2}), audio_data_all_1, 16000);
audiowrite(sprintf('%s/%s_merged_all.wav', foder_to_save, names{3}), audio_data_all_2, 16000);
audiowrite(sprintf('%s/%s_merged_all.wav', foder_to_save, names{4}), audio_data_all_3, 16000);
audiowrite(sprintf('%s/%s_merged_all.wav', foder_to_save, names{5}), audio_data_all_4, 16000);
audiowrite(sprintf('%s/%s_merged_all.wav', foder_to_save, names{6}), audio_data_all_4, 16000);
