from glob import glob



def analyse(file_names):
    score_dict = {}
    for filename in file_names:
        task_name = filename.split("/")[-1]
        total_examples = 0
        masked_examles =0
        total_masked_words = 0
        total_words = 0
        with open(filename) as in_file:
            lines = in_file.readlines()
            total_examples = len(lines)/3
            for idx in range(0,len(lines),3):
                masked_count = 0
                masked_line = lines[idx+1].strip().lower()
                words_count = len(masked_line.split())
                for word in masked_line.split():
                    if word=='[mask]':
                        masked_count +=1
                if masked_count!=0:
                    masked_examles +=1
                total_words += words_count
                total_masked_words += masked_count
        score_dict[task_name] = {"total_example": total_examples,"masked_exampls":masked_examles,"total_words":total_words,"total_masked_words": total_masked_words}
    for task_name in sorted(score_dict.keys()):
        scores = score_dict[task_name]
        print("{}".format(task_name))
        _ = [ print("{}:{}\n".format(k,v)) for k,v in scores.items()]







if __name__=="__main__":
    file_names = glob("./fewshot_mams_1/*")
    analyse(file_names)
