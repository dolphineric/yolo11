from pydub import AudioSegment

# 讀取原始音樂
audio = AudioSegment.from_file("input.mp3")

# 循環三遍
looped_audio = audio * 3

# 輸出新音檔
looped_audio.export("output_loop_3times.mp3", format="mp3")