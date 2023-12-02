class WriteToFile:

    data_names_array = ["Seed", "Liczba warstw neuronów",
                        "Warstwa wejściowa", "Warstwa ukryta",
                        "Warstwa wyjściowa", "Liczba epok"]
    data_array = []
    epochs_array = []
    result = ""
    num_of_layers = 0

    def __init__(self):
        self.file_name = "Lab2_Zad1_Output.txt"

    def write_to_file(self):
        data_array = WriteToFile.convert_data_array(self)
        WriteToFile.write_data(self, data_array)
        WriteToFile.write_epochs(self)
        file = open(self.file_name, "a")
        file.write(self.result)
        file.write("\n\n\n\n")
        file.close()

    def write_epochs(self):
        file = open(self.file_name, "a")
        for iterator in self.epochs_array:
            file.write(iterator + "\n")
        file.close()
        WriteToFile.write_empty(self)

    def write_data(self, array):
        file = open(self.file_name, "a")
        index = 0
        for iterator in array:
            index += 1
            if index >= len(array):
                break
            if index % 2 == 1:
                file.write(str(iterator) + ": " + str(array[index]) + "\n")
        file.close()
        WriteToFile.write_empty(self)

    def write_empty(self):
        file = open(self.file_name, "a")
        file.write("\n")
        file.close()

    def data_update(self, *data):
        for arg in data:
            self.data_array.append(arg)

    def result_update(self, result):
        self.result = result

    def layers_uprate(self, num_of_layers):
        self.num_of_layers = num_of_layers

    def epoch_update(self, epoch_data):
        self.epochs_array.append(epoch_data)

    def convert_data_array(self):
        new_array = []
        second_iterator = 0
        for iterator in range(len(self.data_array)):
            if self.num_of_layers > 3 and iterator == 3:
                new_array.append(WriteToFile.convert_hidden_layers(self))
                second_iterator = second_iterator + self.num_of_layers
                continue
            new_array.append(self.data_names_array[iterator])
            new_array.append(self.data_array[second_iterator])
            second_iterator += 1
        return new_array

    def convert_hidden_layers(self):
        new_arr = []
        for iterator in range(self.num_of_layers):
            new_arr.append(self.data_names_array[3])
            new_arr.append(self.data_array[iterator])
        return new_arr
