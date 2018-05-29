# coding=utf-8
import os
from scipy.io import wavfile
from align.align_tools import load_aligns, load_boundaries


class FeatureGenerator(object):
    def __init__(self, phones_path, wav_dir_path, aligns_path):
        self.phones_path = phones_path
        self.wav_dir_path = wav_dir_path
        self.aligns_path = aligns_path
        self.phones_id = self.__load_phones_id(self.phones_path)
        self.wav_paths = self.__load_wav_paths(self.wav_dir_path)
        self.aligns_dict = load_aligns(self.aligns_path)
        self.bound_dict = load_boundaries(self.aligns_dict)

    def gen_feats(self, win_radius=32):
        """
        Generate features & labels for classifier.
        :param win_radius: sample numbers.
        :return:
        """
        for wav_name in self.bound_dict.keys():
            '''read raw wave'''
            fs, data = wavfile.read(self.wav_paths[wav_name])
            data = list(data)  # trans numpy array to list
            assert fs == 16000
            '''package boundaries'''
            boundaries = self.bound_dict[wav_name]
            for idx, b_tuple in enumerate(boundaries):
                time, l_phone, n_phone, frame_distance = b_tuple
                location = int(time * fs)
                left_location = int(location - frame_distance)\
                    if location - frame_distance >= 0 else 0
                right_location = int(location + frame_distance)\
                    if location + frame_distance <= len(data) else len(data)
                try:
                    '''Left frame'''
                    yield (self.__slice_samples(data, left_location, win_radius),
                           self.phones_id[l_phone], self.phones_id[n_phone], [0])
                    '''Center frame'''
                    yield (self.__slice_samples(data, location, win_radius),
                           self.phones_id[l_phone], self.phones_id[n_phone], [1])
                    '''Right frame'''
                    yield (self.__slice_samples(data, right_location, win_radius),
                           self.phones_id[l_phone], self.phones_id[n_phone], [2])
                except AssertionError:
                    # TODO @yangshuai handle when new location is out of samples range.
                    print('Slice exception!')
                    print(' wav_name: {}, total_sample_num: {}, sample_location: {}.'
                          .format(wav_name, len(data), location))
                    print(' boundary info: {}.'.format(b_tuple))

    def gen_bound_info(self):
        """
        Generate bound_info for align mender.
        :return:
        """
        bound_info = list()  # list: [(align_name, bound_index, time), ...]

        for wav_name in self.bound_dict.keys():
            '''read raw wave'''
            fs, data = wavfile.read(self.wav_paths[wav_name])
            assert fs == 16000
            '''package boundaries'''
            boundaries = self.bound_dict[wav_name]
            for idx, b_tuple in enumerate(boundaries):
                time, l_phone, n_phone, frame_distance = b_tuple
                location = int(time * fs)
                left_location = int(location - frame_distance)\
                    if location - frame_distance >= 0 else 0
                right_location = int(location + frame_distance)\
                    if location + frame_distance <= len(data) else len(data)
                '''Left frame'''
                bound_info.append((wav_name, idx, left_location/fs))
                '''Center frame'''
                bound_info.append((wav_name, idx, time))
                '''Right frame'''
                bound_info.append((wav_name, idx, right_location/fs))

        return bound_info

    def old_gen_feats(self, win_radius=32):
        """
        Generate features & labels for classifier.
        :param win_radius: sample numbers.
        :return:
        """
        samples = list()
        last_phones = list()
        next_phones = list()

        labels = list()
        bound_info = list()  # list: [(align_name, bound_index, time), ...]

        for wav_name in self.bound_dict.keys():
            '''read raw wave'''
            fs, data = wavfile.read(self.wav_paths[wav_name])
            data = list(data)  # trans numpy array to list
            assert fs == 16000
            '''package boundaries'''
            boundaries = self.bound_dict[wav_name]
            for idx, b_tuple in enumerate(boundaries):
                time, l_phone, n_phone, frame_distance = b_tuple
                location = int(time * fs)
                left_location = location - frame_distance
                right_location = location + frame_distance
                '''Left frame'''
                samples.append(self.__slice_samples(data, left_location, win_radius))
                last_phones.append(self.phones_id[l_phone])
                next_phones.append(self.phones_id[n_phone])
                labels.append(0)
                bound_info.append((wav_name, idx, left_location/fs))
                '''Center frame'''
                samples.append(self.__slice_samples(data, location, win_radius))
                last_phones.append(self.phones_id[l_phone])
                next_phones.append(self.phones_id[n_phone])
                labels.append(1)
                bound_info.append((wav_name, idx, time))
                '''Right frame'''
                samples.append(self.__slice_samples(data, right_location, win_radius))
                last_phones.append(self.phones_id[l_phone])
                next_phones.append(self.phones_id[n_phone])
                labels.append(2)
                bound_info.append((wav_name, idx, right_location/fs))

        return {'samples': samples, 'last_phones': last_phones,
                'next_phones': next_phones, 'labels': labels}, bound_info

    def reload_boundaries(self):
        self.aligns_dict = load_aligns(self.aligns_path)
        self.bound_dict = load_boundaries(self.aligns_dict)

    @staticmethod
    def __slice_samples(samples, location, win_radius):
        assert type(location) is int
        frame = samples[location-win_radius: location+win_radius+1]
        assert len(frame) == win_radius*2 + 1
        return frame

    @staticmethod
    def __load_phones_id(phones_path):
        """
        Load phone name and id.
        :param phones_path:
        :param phones_path:
        :return: phones_id
            dict: {name1: id1, name2, id2, ...}
        """
        phones_id = dict()
        count = 0
        with open(phones_path, 'r') as f:
            while 1:
                lines = f.readlines(1000)
                if not lines:
                    break
                for line in lines:
                    phones_id[line.strip()] = count
                    count += 1
        return phones_id

    @staticmethod
    def __load_wav_paths(wav_dir_path):
        """
        Load wav path dict.
        :param wav_dir_path:
        :return: wav_paths
            dict: {name1: path1, name2:path2, ...}
        """
        wav_paths = dict()
        for file_name in os.listdir(wav_dir_path):
            base_name = file_name.split('.')[0]
            wav_paths[base_name] = os.path.join(wav_dir_path, file_name)
        return wav_paths
