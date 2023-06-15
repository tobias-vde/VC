use image::{imageops::FilterType, DynamicImage, GenericImage, GenericImageView, Rgba};
use ndarray::{s, Array, Axis, IxDyn};
use ort::{tensor::InputTensor, Environment, SessionBuilder};

const DIMS: u32 = 640;

fn main() {
    let input_imgage_path = std::env::args().nth(1).unwrap_or("data/img2.png".to_string());
	let buf = std::fs::read(input_imgage_path).unwrap();
	let mut img = image::load_from_memory(&buf).unwrap();
	let (model_input, img_width, img_height) = input_to_tensor(&img);
	let model_output = run_model(model_input);
	let output = output_to_bboxes(model_output, img_width, img_height);
	for (x1, y1, x2, y2, label, probability) in output {
		let a = (probability * 255.0) as u8;
		let rgb = Rgba([255, 0, a, 255]); //TODO: https://crates.io/crates/colorgrad
		const RGBA_RED: Rgba<u8> = Rgba([255, 0, 0, 255]);
		for x in (x1 as u32)..(x2 as u32) {
			img.put_pixel(x as _, y1 as _, RGBA_RED);
			img.put_pixel(x as _, y2 as _, rgb);
		}
		for y in (y1 as u32)..(y2 as u32) {
			img.put_pixel(x1 as _, y, RGBA_RED);
			img.put_pixel(x2 as _, y, rgb);
		}
		let probability_percent = probability * 100.0;
		println!("{label} ~ {probability_percent:.2}% @ ({x1:4.0}, {y1:4.0}), ({x2:4.0}, {y2:4.0})");
	}
	img.save("output.png").unwrap();
	show_image(img);
}

#[allow(unused)]
/// Create window and display image with bboxes
fn show_image(img: DynamicImage) {
	//TODO: OpenCV improve
	use show_image::AsImageView;
	show_image::run_context(move || {
		let win = show_image::create_window("img", show_image::WindowOptions::default()).unwrap();
		//win.run_function_wait(move |mut win| win.set_image("img", &img.as_image_view().unwrap())).unwrap();
		win.set_image("img", img.as_image_view().unwrap()).unwrap();
		win.wait_until_destroyed().unwrap();
	});
	unreachable!("show_image::run_context() returned?");
}

/// Convert input image to YOLOv8 input tensor
/// Returns (original image as input tensor, original image width, original image height)
fn input_to_tensor(img: &DynamicImage) -> (Array<f32, IxDyn>, u32, u32) {
	let (img_width, img_height) = (img.width(), img.height());
	let img = img.resize_exact(DIMS, DIMS, FilterType::CatmullRom);
	let mut input = Array::zeros((1, 3, DIMS as usize, DIMS as usize)).into_dyn();
	for pixel in img.pixels() {
		let x = pixel.0 as usize;
		let y = pixel.1 as usize;
		let [r, g, b, _] = pixel.2 .0;
		input[[0, 0, y, x]] = (r as f32) / 255.0;
		input[[0, 1, y, x]] = (g as f32) / 255.0;
		input[[0, 2, y, x]] = (b as f32) / 255.0;
	}
	return (input, img_width, img_height);
}

/// Runs YOLOv8 with given input tensor.
/// Returns output of YOLOv8 network as a single dimension array
fn run_model(input: Array<f32, IxDyn>) -> Array<f32, IxDyn> {
	let env = std::sync::Arc::new(Environment::builder().with_name("YOLOv8").build().unwrap());
	let model = SessionBuilder::new(&env)
		.unwrap()
		.with_model_from_file("yolov8m.onnx")
		//.with_model_downloaded(ort::download::vision::object_detection_image_segmentation::ObjectDetectionImageSegmentation::YoloV4)
		.unwrap();
	let input = InputTensor::FloatTensor(input);
	let outputs = model.run([input]).unwrap();
	let output = outputs
		.get(0)
		.unwrap()
		.try_extract::<f32>()
		.unwrap()
		.view()
		.t() //YoloV4?
		.into_owned();
	return output;
}

/// Convert YOLOv8 raw output array to an array of detected objects.
/// Returns array of detected objects as vec [(x1,y1, x2,y2, label, probability), ..]
fn output_to_bboxes(
	output: Array<f32, IxDyn>,
	img_width: u32,
	img_height: u32,
) -> Vec<(f32, f32, f32, f32, &'static str, f32)> {
	let mut boxes = Vec::new(); //TODO: map-filter-{find?}-reduce-collect-sort
	let output = output.slice(s![.., .., 0]); //2D
	for row in output.axis_iter(Axis(0)) {
		let row: Vec<_> = row.iter().map(|x| *x).collect(); // deref values...
		let (class_id, probability) = row
			.iter()
			.skip(4)
			.enumerate()
			.map(|(index, value)| (index, *value))
			.reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
			.unwrap(); // with best probability: find()?
		if probability < 0.5 {
			continue;
		}
		let dims_f32 = DIMS as f32;
		let xc = (row[0] / dims_f32) * (img_width as f32);
		let yc = (row[1] / dims_f32) * (img_height as f32);
		let w = (row[2] / dims_f32) * (img_width as f32);
		let h = (row[3] / dims_f32) * (img_height as f32);
		let x1 = xc - (w / 2.0);
		let x2 = xc + (w / 2.0);
		let y1 = yc - (h / 2.0);
		let y2 = yc + (h / 2.0);
		let label = YOLO_CLASSES[class_id];
		boxes.push((x1, y1, x2, y2, label, probability));
	}

	boxes.sort_by(|a, b| b.5.total_cmp(&a.5)); //sorts by prob
	let mut result = Vec::new(); // only keep boxes that are not overlapping (iou) the first one
	const MAX_IOU_THRESHOLD: f32 = 0.7;
	while boxes.len() > 0 {
		result.push(boxes[0]);
		boxes = boxes
			.iter()
			.filter(|box1| iou(&boxes[0], box1) < MAX_IOU_THRESHOLD)
			.map(|x| *x)
			.collect();
	}
	return result;
}

/// Returns Intersection over union ratio as a float number
// https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/.
fn iou(
	a: &(f32, f32, f32, f32, &'static str, f32),
	b: &(f32, f32, f32, f32, &'static str, f32),
) -> f32 {
	return intersection(a, b) / union(a, b);
}

/// Returns Area of the union
fn union(
	a: &(f32, f32, f32, f32, &'static str, f32),
	b: &(f32, f32, f32, f32, &'static str, f32),
) -> f32 {
	let (a_x1, a_y1, a_x2, a_y2, _, _) = *a;
	let (b_x1, b_y1, b_x2, b_y2, _, _) = *b;
	let a_area = (a_x2 - a_x1) * (a_y2 - a_y1);
	let b_area = (b_x2 - b_x1) * (b_y2 - b_y1);
	return (a_area + b_area) - intersection(a, b);
}

/// Returns Area of intersection
fn intersection(
	a: &(f32, f32, f32, f32, &'static str, f32),
	b: &(f32, f32, f32, f32, &'static str, f32),
) -> f32 {
	let (a_x1, a_y1, a_x2, a_y2, _, _) = *a;
	let (b_x1, b_y1, b_x2, b_y2, _, _) = *b;
	let x1 = a_x1.max(b_x1);
	let y1 = a_y1.max(b_y1);
	let x2 = a_x2.min(b_x2);
	let y2 = a_y2.min(b_y2);
	return (x2 - x1) * (y2 - y1);
}

// YOLOv8 class labels (label num -> label &str)
const YOLO_CLASSES: [&str; 80] = [
	"person",
	"bicycle",
	"car",
	"motorcycle",
	"airplane",
	"bus",
	"train",
	"truck",
	"boat",
	"traffic light", // the only one we care about...
	"fire hydrant",
	"stop sign",
	"parking meter",
	"bench",
	"bird",
	"cat",
	"dog",
	"horse",
	"sheep",
	"cow",
	"elephant",
	"bear",
	"zebra",
	"giraffe",
	"backpack",
	"umbrella",
	"handbag",
	"tie",
	"suitcase",
	"frisbee",
	"skis",
	"snowboard",
	"sports ball",
	"kite",
	"baseball bat",
	"baseball glove",
	"skateboard",
	"surfboard",
	"tennis racket",
	"bottle",
	"wine glass",
	"cup",
	"fork",
	"knife",
	"spoon",
	"bowl",
	"banana",
	"apple",
	"sandwich",
	"orange",
	"broccoli",
	"carrot",
	"hot dog",
	"pizza",
	"donut",
	"cake",
	"chair",
	"couch",
	"potted plant",
	"bed",
	"dining table",
	"toilet",
	"tv",
	"laptop",
	"mouse",
	"remote",
	"keyboard",
	"cell phone",
	"microwave",
	"oven",
	"toaster",
	"sink",
	"refrigerator",
	"book",
	"clock",
	"vase",
	"scissors",
	"teddy bear",
	"hair drier",
	"toothbrush",
];
