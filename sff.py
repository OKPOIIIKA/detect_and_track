import os
import struct
from PIL import Image
import numpy as np
from typing import Optional, Tuple, Generator
import io
import cv2


class SFFReader:
    """
    Класс для чтения и обработки данных из файлов формата SFF."""

    def __init__(self, sff_file: str, dat_file: str = None):
        """
        Инициализирует объект SFFReader.

        Args:
            sff_file: Путь к файлу .sff.
            dat_file: Путь к индексному файлу .dat (опционально).
                      Если не указан, предполагается, что он имеет то же имя,
                      что и файл .sff, но с расширением .dat.
        """
        self.sff_file = sff_file
        self.dat_file = dat_file or sff_file.replace('.sff', '.dat')
        self.header = self._read_header()
        self.frame_data = self._read_dat()  # Читаем данные о кадрах из DAT
        self.frame_count = len(self.frame_data)

        # Проверка и корекция пикетов
        self.recalculate_pickets()

    def _read_header(self) -> dict:
        """
        Читает и разбирает заголовок файла .sff.

        Returns:
            Словарь с данными из заголовка.
        """
        with open(self.sff_file, 'rb') as f:
            header = {
                'road_code': struct.unpack('<h', f.read(2))[0], # код дороги 
                'road_name_length': struct.unpack('<h', f.read(2))[0], # длина названия дороги
                'road_name': f.read(200).decode('cp1251').rstrip('\x00'), # название дороги
                'direction': struct.unpack('<h', f.read(2))[0], # направление
                'recording_date': struct.unpack('<d', f.read(8))[0], # дата записи
                'start_km': struct.unpack('<d', f.read(8))[0], # начало отрезка
                'end_km': struct.unpack('<d', f.read(8))[0], # конец отрезка
                'reserved': f.read(5), # резерв
            }
        return header

    def _read_dat(self) -> list:
        """
        Читает информацию о кадрах из DAT-файла.

        Returns:
            Список словарей с информацией о каждом кадре.
        """
        frame_data = []

        with open(self.dat_file, 'rb') as f_dat:
            while True:
                dat_data = f_dat.read(24)  # Читаем данные из DAT-файла
                if len(dat_data) < 24:
                    break
                jpeg_size, offset, picket, timestamp = struct.unpack('<iqid', dat_data)

                frame_data.append({
                    'jpeg_size': jpeg_size,
                    'offset': offset,
                    'picket': picket,
                    'timestamp': timestamp,
                })

        return frame_data

    def get_frames_count(self):
        return self.frame_count

    def get_frame_by_number(self, frame_number: int, as_bytes: bool = False):
        """
        Возвращает изображение кадра или его байты по номеру, используя смещение из DAT файла.

        Args:
            frame_number: Номер кадра.
            as_bytes: Если True — вернуть байты JPEG, иначе — PIL Image.

        Returns:
            PIL Image или bytes: Изображение или байты JPEG.
        """
        if frame_number >= self.frame_count:
            raise ValueError(f"Номер кадра {frame_number} превышает количество кадров в видео.")

        frame_info = self.frame_data[frame_number]
        offset = frame_info['offset']
        with open(self.sff_file, 'rb') as f:
            f.seek(offset)
            jpeg_size = frame_info['jpeg_size']
            jpeg_data = f.read(jpeg_size).rstrip(b'\x00')
        if as_bytes:
            return jpeg_data
        else:
            return Image.open(io.BytesIO(jpeg_data))
    
    def get_frame_by_meter(self, meter: int) -> Tuple[Optional[Image.Image], int]:
        """
        Возвращает кадр, ближайший к заданному значению метра, 
        и его номер в видео.

        Args:
            meter: Значение метра.

        Returns:
            Tuple[Optional[Image.Image], int]: Кортеж, содержащий:
                - Изображение кадра (PIL Image) или None, если кадр не найден.
                - Номер кадра.
        """
        closest_frame_num = None
        min_distance = float('inf')
        for i, frame_data in enumerate(self.frame_data):
            distance = abs(frame_data['picket'] - meter)
            if distance < min_distance:
                closest_frame_num = i
                min_distance = distance

        if closest_frame_num is not None:
            return self.get_frame_by_number(closest_frame_num), closest_frame_num
        else:
            return None, -1
        
    def get_frames(self, direction: int=0, start_km: float=None, end_km:float=None,
                   reverse: bool = False, picket_thr: int = 2  
                   ) -> Generator[Tuple[int, Image.Image, int, float], None, None]:
        """
        Генератор, возвращающий кадры из видео SFF.

        Args:
            direction: Направление проезда.
            start_km: Начало отрезка дороги для обработки.
            end_km: Конец отрезка дороги для обработки.
            reverse: Флаг, указывающий, нужно ли инвертировать порядок кадров.
            picket_thr: Минимальная разница в пикетах (в метрах)

        Yields:
            Tuple[Image.Image, int, float]: Кортеж, содержащий:
                - Номер кадра.
                - Кадр изображения (PIL Image).
                - Значение пикета.
                - Временную метку кадра.
        """

        previous_picket = None
        prev_frame_skipped = False # Флаг пропуска кадра

        with open(self.sff_file, 'rb') as f:
            frame_indices = list(range(len(self.frame_data))) #Создаем список индексов

            if reverse:
                frame_indices = frame_indices[::-1]  # Инвертируем список индексов, если reverse=True

            for frame_num in frame_indices:
                frame_info = self.frame_data[frame_num]
                km = frame_info['picket'] / 1000

                if not reverse: # Переднее расположение камер
                    if direction == 0:  # Прямое направление
                        if start_km is not None and km < start_km:
                            continue
                        if end_km is not None and km > end_km:
                            break
                    else:  # Обратное направление
                        if start_km is not None and km > start_km:
                            continue
                        if end_km is not None and km < end_km:
                            break
                else: # Заднее расположение камер
                    if direction == 0:  # Прямое направление
                        if start_km is not None and km > end_km:
                            continue
                        if end_km is not None and km < start_km:
                            break
                    else:  # Обратное направление
                        if start_km is not None and km < end_km:
                            continue
                        if end_km is not None and km > start_km:
                            break

                # Пропуск кадров в зависимости от расстояния
                current_picket = frame_info['picket']
                if previous_picket is not None:
                    picket_difference = abs(current_picket - previous_picket)
                    if picket_difference < picket_thr and not prev_frame_skipped:
                        prev_frame_skipped = True
                        continue
                    else:
                        prev_frame_skipped = False
                previous_picket = current_picket

                f.seek(frame_info['offset'])
                jpeg_size = frame_info['jpeg_size']
                jpeg_data = f.read(jpeg_size).rstrip(b'\x00')
                image = Image.open(io.BytesIO(jpeg_data))
                yield frame_num, image, frame_info['picket'], frame_info['timestamp']

    def create_sff_and_dat(self, output_folder: str,
                           frames_generator: Generator[Tuple[Image.Image, int, float], None, None]):
        """
        Создает файлы .sff и .dat, записывая кадры по одному.

        Args:
            output_folder: Путь к папке для сохранения выходных файлов.
            frames_generator: Генератор, возвращающий кортежи (frame, picket, float).
        """
        sff_file = os.path.join(output_folder, 'Video.sff')
        dat_file = os.path.join(output_folder, 'Video.dat')

        os.makedirs(output_folder, exist_ok=True)

        with open(sff_file, 'wb') as f_sff, open(dat_file, 'wb') as f_dat:
            # Записываем заголовок SFF из исходного файла
            f_sff.write(struct.pack('<h', self.header['road_code']))
            f_sff.write(struct.pack('<h', self.header['road_name_length']))
            f_sff.write(self.header['road_name'].encode('cp1251').ljust(200, b'\x00'))
            f_sff.write(struct.pack('<h', self.header['direction']))
            f_sff.write(struct.pack('<d', self.header['recording_date']))
            f_sff.write(struct.pack('<d', self.header['start_km']))
            f_sff.write(struct.pack('<d', self.header['end_km']))
            f_sff.write(self.header['reserved'])

            offset = 239  # Начальное смещение для первого кадра
            frame_num = 0

            for frame, picket, timestamp in frames_generator:
                # Сохраняем кадр в JPEG байты
                output_buffer = io.BytesIO()
                frame.save(output_buffer, format='JPEG')
                jpeg_data = output_buffer.getvalue()
                jpeg_size = len(jpeg_data)

                # Записываем данные кадра в SFF
                f_sff.write(struct.pack('<i', jpeg_size))
                f_sff.write(jpeg_data)
                f_sff.write(struct.pack('<i', picket))
                f_sff.write(struct.pack('<d', timestamp))

                # Записываем данные кадра в DAT
                f_dat.write(struct.pack('<i', jpeg_size))
                f_dat.write(struct.pack('<q', offset))
                f_dat.write(struct.pack('<i', picket))
                f_dat.write(struct.pack('<d', timestamp))

                offset += jpeg_size + 16  # Обновляем смещение для следующего кадра
                frame_num += 1

            # Записываем общее количество кадров в SFF
            f_sff.write(struct.pack('<i', frame_num))

    def recalculate_pickets(self):
        """
        Проверяет и пересчитывает пикеты в frame_data, если есть расхождение между заголовком SFF и данными DAT.
        """
        start_m_header = self.header.get('start_km', 0) * 1000
        end_m_header = self.header.get('end_km', 0) * 1000

        start_m_dat = self.frame_data[0]['picket']
        end_m_dat = self.frame_data[-1]['picket']

        if start_m_dat == end_m_dat:
            print("[!] Все кадры имеют одинаковый пикет — пересчет невозможен.")
            return

        if start_m_header == 0 and end_m_header == 0:
            self.header['start_km'] = round(start_m_dat / 1000, 3)
            self.header['end_km'] = round(end_m_dat / 1000, 3)

        if start_m_header != start_m_dat or end_m_header != end_m_dat:
            for frame_info in self.frame_data:
                current_m_dat = frame_info['picket']
                picket_offset = (end_m_header - start_m_header) * (current_m_dat - start_m_dat) / (end_m_dat - start_m_dat)
                new_picket = start_m_header + picket_offset
                frame_info['picket'] = int(round(new_picket))

def save_sff_to_mp4(sff_path: str, output_path: str, fps: int = 25, resize: Optional[Tuple[int, int]] = None):
    reader = SFFReader(sff_path)
    print("Чтение заголовка:", reader.header)
    
    # Получаем первый кадр, чтобы определить размер
    try:
        first_result = next(reader.get_frames())
    except StopIteration:
        raise RuntimeError("Нет кадров в SFF-файле.")

    frame_num, first_frame, picket, timestamp = first_result

    if first_frame is None:
        raise ValueError("Первый кадр не загружен (None).")

    first_frame_np = np.array(first_frame)

    if first_frame_np.size == 0:
        raise ValueError("Первый кадр пуст (некорректный JPEG или сбой при чтении).")
    
    if resize:
        width, height = resize
    else:
        height, width = first_frame_np.shape[:2]

    # Создаем видеопишущий объект
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Сохранение кадров в видео...")
    frame_count = 0
    for _, frame, picket, timestamp in reader.get_frames():
        frame_np = np.array(frame)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        if resize:
            frame_bgr = cv2.resize(frame_bgr, (width, height))

        out.write(frame_bgr)
        frame_count += 1

    out.release()
    print(f"Готово. Сохранено {frame_count} кадров в {output_path}")

if __name__ == '__main__':
    # Путь к файлу SFF
    sff_file = r'D:\учеба\Практика\расчет_интенсивности\мониторинг_видео\Нефтеюганск_2023_ул.Набережная-Ленина\Camera_2\Video.sff'
    # output_mp4 = r'D:\учеба\Практика\расчет_интенсивности\мониторинг_видео\Camera_2_output.mp4'

    # save_sff_to_mp4(sff_file, output_mp4, fps=30)

    # Создаем экземпляр SFFReader
    reader = SFFReader(sff_file)

    # Вывод информации из заголовка
    print(reader.header)

    # Получаем количество кадров
    frame_count = reader.get_frames_count()
    print(f"Total frames: {frame_count}")

    # Настройки для вывода видео
    fps = 30  # Кадров в секунду (можно настроить)
    delay_ms = int(1000 / fps)  # Задержка между кадрами в миллисекундах

    # Цикл для отображения видео
    for frame_num, image, picket, timestamp in reader.get_frames():
        # Преобразуем PIL Image в NumPy массив
        image_np = np.array(image)

        # Преобразуем изображение из RGB в BGR (формат OpenCV)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Отображаем кадр
        cv2.imshow('Video Playback', image_bgr)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Очищаем окна OpenCV
    cv2.destroyAllWindows()
    