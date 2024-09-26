mod dynamic_array;

use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::io::Read;
use std::ops::Index;
use std::sync::Arc;
use std::thread::sleep;
use std::time::Instant;
use arrayfire::*;
use indexmap::IndexMap;
use rand::distr::Uniform;
use rand::prelude::ThreadRng;

extern crate rand;

use rand::Rng;
use sendable::SendRc;
use typed_arena::Arena;
use crate::dynamic_array::DynamicArray;

fn main() {
    // Capture the command-line arguments
    let args: Vec<String> = env::args().collect();

    // Check if the user provided a command-line argument for which main function to run
    if args.len() < 2 {
        eprintln!("Usage: {} <main_method>", args[0]);
        eprintln!("Available main methods: main123, mainx, main_x");
        return;
    }
    let arena = Arena::new();
    let arena2 = Arena::new();
    let arena3 = Arena::new();
    let arena4 = Arena::new();


    // Match the argument to the corresponding main function
    match args[1].as_str() {
        "flat_array_main" => flat_array_main(),
        "nested_array_main" => nested_array_main(),
        "gpu_main" => gpu_main(),
        "chunked_table_main" => test_harness("chunked_table", |num_rows, num_cols| ChunkedTable::new(num_rows, num_cols)),
        "arena_row_chunked_table_main" => test_harness("arena_row_chunked_table", |num_rows, _| {
            ArenaRowChunkedTable::new(&arena, num_rows)
        }),
        "arena_chunked_table_main" => test_harness("arena_chunked_table", |num_rows, num_cols| {
            ArenaChunkedTable::new(&arena2, num_rows, num_cols)
        }),
        "main_copy_data" => main_copy_data(),
        "main_copy_data2" => main_copy_data2(),
        "arena_row_chunked_table_ref_cell_main" => test_harness("arena_row_chunked_table_ref_cell", |num_rows, _| {
            let arena = Arena::new();
            ArenaRowChunkedTableRefCell::new(&arena, num_rows)
        }),
        "arena_row_chunked_table_raw_pointer_main" => test_harness("arena_row_chunked_table_raw_pointer", |num_rows, _| {
            ArenaRowChunkedTableRawPointer::new(&arena, num_rows)
        }),
        "arena_row_chunked_table_raw_pointer_dynamic_cols_main" => test_harness("arena_row_chunked_table_raw_pointer_dynamic_cols", |num_rows, _| {
            ArenaRowChunkedTableRawPointerDynamicCols::new(&arena3, num_rows,3)
        }),
        "arena_row_chunked_table_raw_pointer_dynamic_array_main" => test_harness("arena_row_chunked_table_raw_pointer_dynamic_array", |num_rows, _| {
            ArenaRowChunkedTableRawPointerDynamicArray::new(&arena4, num_rows,3)
        }),
        "arena_row_chunked_table_arc_main" => test_harness("arena_row_chunked_table_arc", |num_rows, _| {
            let arena = Arena::new();
            ArenaRowChunkedTableSendRc::new(&arena, num_rows)
        }),
        "arena_row_chunked_table_soa_main" => test_harness("arena_row_chunked_table_soa", |num_rows, _| {
            ArenaRowChunkedTableSoA::new(num_rows)
        }),
        "arena_row_chunked_table_single_static_array" => test_harness("arena_row_chunked_table_single_static_array", |num_rows, num_cols| {
            ArenaStaticArrayRowChunkedTable::new(Arena::new(),num_rows, num_cols)
        }),
        "main_recell_arena" => main_recell_arena(),
        "main_arc_send" => main_arc_send(),
        _ => {
            eprintln!("Invalid option: {}. Use one of: flat_array_main, nested_array_main, gpu_main,chunked_table_main", args[1]);
        }
    }
}


fn main_arc_send() {
    // Create a new arena for chunks
    let arena = Arena::new();

    // Create a table with 10,000 rows
    let mut table = ArenaRowChunkedTableSendRc::new(&arena, 10_000);

    // Clone the table so that it can be shared across threads

    // Spawn a thread to work with the arena-allocated resource
    let handle = std::thread::spawn(move || {
        // Insert values into the table in the new thread
        table.insert_value(0, 0, 1.0);
        table.insert_value(0, 1, 2.0);
        table.insert_value(0, 2, 3.0);

        // Retrieve and print the values
        println!("Thread: Row 0, Col 0: {}", table.get_value(0, 0));
        println!("Thread: Row 0, Col 1: {}", table.get_value(0, 1));
        println!("Thread: Row 0, Col 2: {}", table.get_value(0, 2));
    });

    handle.join().unwrap();
    //
    // // Access the table from the main thread
    // println!("Main: Row 0, Col 0: {}", table.get_value(0, 0));
    // println!("Main: Row 0, Col 1: {}", table.get_value(0, 1));
    // println!("Main: Row 0, Col 2: {}", table.get_value(0, 2));
}


struct MyStruct {
    a: u64,
    b: i32,
    c: i64,
    d: i8,
}

fn main_recell_arena() {
    // Start timing the entire process
    let overall_start = Instant::now();

    // Create an arena to hold `MyStruct` instances
    let arena = Arena::new();

    // Start timing the allocation
    let alloc_start = Instant::now();

    let num_rows: usize = 100_000_000;

    // Allocate 1 million `MyStruct` objects in the arena
    let mut objects = Vec::with_capacity(num_rows);
    for _ in 0..num_rows {
        let obj = arena.alloc(MyStruct {
            a: rand::random::<u64>(),
            b: rand::random::<i32>(),
            c: rand::random::<i64>(),
            d: rand::random::<i8>(),
        });
        objects.push(obj);
    }

    // End timing the allocation
    let alloc_duration = alloc_start.elapsed();
    println!("Time to allocate 1 million objects: {:?}", alloc_duration);

    // Start timing the summation
    let sum_start = Instant::now();

    // Sum the fields `a`, `b`, `c`, and `d` for all objects
    let mut total_a: u64 = 0;
    let mut total_b: i32 = 0;
    let mut total_c: i64 = 0;
    let mut total_d: i8 = 0;

    for obj in &objects {
        total_a += obj.a;
        total_b += obj.b;
        total_c += obj.c;
        total_d += obj.d;
    }

    // End timing the summation
    let sum_duration = sum_start.elapsed();
    println!("Time to sum the values of 1 million objects: {:?}", sum_duration);

    // Print total sums
    println!("Total a: {}", total_a);
    println!("Total b: {}", total_b);
    println!("Total c: {}", total_c);
    println!("Total d: {}", total_d);

    // End overall timing
    let overall_duration = overall_start.elapsed();
    println!("Total time for allocation and summation: {:?}", overall_duration);
}


const CHUNK_SIZE: usize = 1024; // Number of rows per chunk

// Structure to hold chunked data for each column
struct Column {
    chunks: Vec<Vec<f32>>, // Each chunk stores a fixed number of rows
}

struct ChunkedTable {
    columns: Vec<Column>,  // List of columns, each containing chunked data
    num_rows: usize,       // Total number of rows
}

impl ChunkedTable {
    fn new(num_rows: usize, num_cols: usize) -> Self {
        let mut columns = Vec::with_capacity(num_cols);
        let num_chunks = (num_rows + CHUNK_SIZE - 1) / CHUNK_SIZE; // Calculate number of chunks needed
        for _ in 0..num_cols {
            // Initialize chunks for each column
            let mut column = Column { chunks: Vec::with_capacity(num_chunks) };
            for _ in 0..num_chunks {
                column.chunks.push(vec![0.0; CHUNK_SIZE]); // Allocate chunk space
            }
            columns.push(column);
        }
        ChunkedTable { columns, num_rows }
    }


}

impl IChunkedTable for ChunkedTable {
    // Insert a value into the chunked table
    fn insert_value(&mut self, row: usize, col: usize, value: f32) {
        let chunk_index = row / CHUNK_SIZE;
        let row_index = row % CHUNK_SIZE;
        self.columns[col].chunks[chunk_index][row_index] = value;
    }

    // Get a value from the chunked table
    fn get_value(&self, row: usize, col: usize) -> f32 {
        let chunk_index = row / CHUNK_SIZE;
        let row_index = row % CHUNK_SIZE;
        self.columns[col].chunks[chunk_index][row_index]
    }

    fn add_column(&mut self) {
        todo!()
    }
}

struct ArenaColumn<'a> {
    chunks: Vec<&'a [f32]>, // Each chunk stores a fixed number of rows
}

struct ArenaChunkedTable<'a> {
    columns: Vec<ArenaColumn<'a>>,  // List of columns, each containing chunked data
    num_rows: usize,           // Total number of rows
}


impl<'a> ArenaChunkedTable<'a> {
    fn new(arena: &'a Arena<[f32; CHUNK_SIZE]>, num_rows: usize, num_cols: usize) -> Self {
        let mut columns = Vec::with_capacity(num_cols);
        let num_chunks = (num_rows + CHUNK_SIZE - 1) / CHUNK_SIZE; // Calculate number of chunks needed
        for _ in 0..num_cols {
            // Initialize chunks for each column
            let mut column = ArenaColumn { chunks: Vec::with_capacity(num_chunks) };
            for _ in 0..num_chunks {
                column.chunks.push(arena.alloc([0.0; CHUNK_SIZE])); // Allocate chunk space using arena
            }
            columns.push(column);
        }
        ArenaChunkedTable { columns, num_rows }
    }


}

impl<'a> crate::IChunkedTable for ArenaChunkedTable<'a> {
    // Insert a value into the chunked table
    fn insert_value(&mut self, row: usize, col: usize, value: f32) {
        let chunk_index = row / CHUNK_SIZE;
        let row_index = row % CHUNK_SIZE;
        let chunk = &mut self.columns[col].chunks[chunk_index];
        unsafe {
            let ptr = chunk.as_ptr() as *mut f32;
            *ptr.add(row_index) = value;
        }
    }

    // Get a value from the chunked table
    fn get_value(&self, row: usize, col: usize) -> f32 {
        let chunk_index = row / CHUNK_SIZE;
        let row_index = row % CHUNK_SIZE;
        self.columns[col].chunks[chunk_index][row_index]
    }

    fn add_column(&mut self) {
        todo!()
    }
}

struct ArenaRowChunkedTable<'a> {
    chunks: Vec<&'a mut [[f32; 3]; CHUNK_SIZE]>, // Each chunk stores CHUNK_SIZE rows, each row has 3 columns
    num_rows: usize,
}

impl<'a> ArenaRowChunkedTable<'a> {
    fn new(arena: &'a Arena<[[f32; 3]; CHUNK_SIZE]>, num_rows: usize) -> Self {
        let num_chunks = (num_rows + CHUNK_SIZE - 1) / CHUNK_SIZE; // Calculate number of chunks needed

        let mut chunks = Vec::with_capacity(num_chunks);
        for _ in 0..num_chunks {
            chunks.push(arena.alloc([[0.0; 3]; CHUNK_SIZE])); // Allocate space for rows, each row has 3 columns
        }

        ArenaRowChunkedTable { chunks, num_rows }
    }

}

impl<'a> IChunkedTable for ArenaRowChunkedTable<'a>  {

    // Insert a value into the chunked table
    fn insert_value(&mut self, row: usize, col: usize, value: f32) {
        let chunk_index = row / CHUNK_SIZE;
        let row_index = row % CHUNK_SIZE;
        self.chunks[chunk_index][row_index][col] = value;
    }

    // Get a value from the chunked table
    fn get_value(&self, row: usize, col: usize) -> f32 {
        let chunk_index = row / CHUNK_SIZE;
        let row_index = row % CHUNK_SIZE;
        self.chunks[chunk_index][row_index][col]
    }

    fn add_column(&mut self) {
        todo!()
    }
}


struct ArenaRowChunkedTableRefCell {
    // Vec of RefCell holding mutable references to each chunk
    chunks: Vec<RefCell<[[f32; 3]; CHUNK_SIZE]>>,
    num_rows: usize,
}

impl ArenaRowChunkedTableRefCell {
    // No lifetimes are needed since RefCell takes care of interior mutability
    fn new(arena: &Arena<[[f32; 3]; CHUNK_SIZE]>, num_rows: usize) -> Self {
        let num_chunks = (num_rows + CHUNK_SIZE - 1) / CHUNK_SIZE; // Calculate number of chunks needed

        let mut chunks = Vec::with_capacity(num_chunks);
        for _ in 0..num_chunks {
            // Allocate space for rows, each row has 3 columns and wrap it in RefCell for mutability
            chunks.push(RefCell::new(*arena.alloc([[0.0; 3]; CHUNK_SIZE])));
        }

        ArenaRowChunkedTableRefCell { chunks, num_rows }
    }
}


pub struct ArenaRowChunkedTableRawPointer {
    chunks: Vec<*mut [[f32; 3]; CHUNK_SIZE]>,  // Vec of raw pointers to each chunk
    num_rows: usize,
    arena: *const Arena<[[f32; 3]; CHUNK_SIZE]>,  // Raw pointer to the arena
}

impl ArenaRowChunkedTableRawPointer {
    // Create a new table that references the arena for memory allocation
    pub fn new(arena: &Arena<[[f32; 3]; CHUNK_SIZE]>, num_rows: usize) -> Self {

        ArenaRowChunkedTableRawPointer {
            chunks: vec![],
            num_rows,
            arena,
        }
    }

    // Dynamically add a chunk if it doesn't exist
    fn add_chunk_if_needed(&mut self, chunk_index: usize) {
        if chunk_index >= self.chunks.len() {
            // Allocate space for a new chunk using the arena
            unsafe {
                let arena = &*self.arena;
                let chunk = arena.alloc([[0.0; 3]; CHUNK_SIZE]);  // Allocate chunk in the arena
                self.chunks.push(chunk as *mut _);  // Store raw pointer to the newly allocated chunk
            }
        }
    }
}

impl IChunkedTable for ArenaRowChunkedTableRawPointer {
    // Insert a value into the table at a specific row and column
    fn insert_value(&mut self, row: usize, col: usize, value: f32) {
        let chunk_index = row / CHUNK_SIZE;
        let row_index = row % CHUNK_SIZE;

        // Ensure the chunk exists before inserting
        self.add_chunk_if_needed(chunk_index);

        unsafe {
            let chunk = self.chunks[chunk_index];
            (*chunk)[row_index][col] = value;  // Dereference raw pointer to access and modify the value
        }

        // Update the number of rows if the new row is beyond the current count
        if row >= self.num_rows {
            self.num_rows = row + 1;
        }
    }

    // Get a value from the table at a specific row and column
    fn get_value(&self, row: usize, col: usize) -> f32 {
        let chunk_index = row / CHUNK_SIZE;
        let row_index = row % CHUNK_SIZE;

        unsafe {
            let chunk = self.chunks[chunk_index];
            (*chunk)[row_index][col]  // Dereference raw pointer to access the value
        }
    }

    // Placeholder for adding a column (not implemented here)
    fn add_column(&mut self) {
        todo!()
    }
}

impl IChunkedTable for ArenaRowChunkedTableRefCell {
    // Insert a value into the chunked table using RefCell borrow_mut for mutation
    fn insert_value(&mut self, row: usize, col: usize, value: f32) {
        let chunk_index = row / CHUNK_SIZE;
        let row_index = row % CHUNK_SIZE;
        // Borrow the chunk mutably and modify the value
        self.chunks[chunk_index].borrow_mut()[row_index][col] = value;
    }

    // Get a value from the chunked table using RefCell borrow for immutably borrowing
    fn get_value(&self, row: usize, col: usize) -> f32 {
        let chunk_index = row / CHUNK_SIZE;
        let row_index = row % CHUNK_SIZE;
        // Borrow the chunk immutably and retrieve the value
        self.chunks[chunk_index].borrow()[row_index][col]
    }

    fn add_column(&mut self) {
        todo!()
    }
}

// Define the struct for the ArenaRowChunkedTable
pub struct ArenaRowChunkedTableRawPointerDynamicCols {
    chunks: Vec<*mut Vec<f32>>,  // Vec of raw pointers to Vec<f32>, each chunk holds rows in contiguous memory
    num_rows: usize,
    num_cols: usize,             // Dynamic number of columns in each row
    arena: *const Arena<Vec<f32>>,  // Raw pointer to the arena
}

impl ArenaRowChunkedTableRawPointerDynamicCols {
    // Create a new table that references the arena for memory allocation
    pub fn new(arena: &Arena<Vec<f32>>, num_rows: usize, num_cols: usize) -> Self {
        let num_chunks = (num_rows + CHUNK_SIZE - 1) / CHUNK_SIZE;  // Calculate number of chunks needed
        let chunks = Vec::with_capacity(num_chunks);  // Start with an empty Vec for chunks

        ArenaRowChunkedTableRawPointerDynamicCols {
            chunks,
            num_rows,
            num_cols,
            arena,
        }
    }

    // Dynamically add a chunk if it doesn't exist
    fn add_chunk_if_needed(&mut self, chunk_index: usize) {
        if chunk_index >= self.chunks.len() {
            // Allocate space for a new chunk using the arena
            unsafe {
                let arena = &*self.arena;
                // Allocate contiguous memory for rows (CHUNK_SIZE rows, each with num_cols columns)
                let chunk: Vec<f32> = vec![0.0; CHUNK_SIZE * self.num_cols];
                self.chunks.push(arena.alloc(chunk) as *mut _);  // Store raw pointer to the newly allocated chunk
            }
        }
    }
}

// Implementation of the IChunkedTable trait for ArenaRowChunkedTable
impl IChunkedTable for ArenaRowChunkedTableRawPointerDynamicCols {
    // Insert a value into the specific column index and row
    fn insert_value(&mut self, row: usize, col_idx: usize, value: f32) {
        let chunk_index = row / CHUNK_SIZE;
        let row_index = row % CHUNK_SIZE;

        // Ensure the chunk exists before inserting
        self.add_chunk_if_needed(chunk_index);

        unsafe {
            let chunk = self.chunks[chunk_index];
            let pos = row_index * self.num_cols + col_idx;  // Direct inline calculation
            (*chunk)[pos] = value;  // Dereference raw pointer to access and modify the value
        }

        // Update the number of rows if the new row is beyond the current count
        if row >= self.num_rows {
            self.num_rows = row + 1;
        }
    }

    // Get a value from the specific column index and row
    fn get_value(&self, row: usize, col_idx: usize) -> f32 {
        let chunk_index = row / CHUNK_SIZE;
        let row_index = row % CHUNK_SIZE;

        unsafe {
            let chunk = self.chunks[chunk_index];
            let pos = row_index * self.num_cols + col_idx;  // Direct inline calculation
            (*chunk)[pos]  // Dereference raw pointer to access the value
        }
    }

    // Add a new column dynamically
    fn add_column(&mut self) {
        self.num_cols += 1;  // Increase the number of columns

        // Reallocate each chunk to accommodate the new column
        for chunk in &mut self.chunks {
            unsafe {
                let old_chunk = &**chunk;  // Dereference raw pointer to get the old chunk
                let mut new_chunk: Vec<f32> = vec![0.0; CHUNK_SIZE * self.num_cols];  // Allocate new chunk

                // Copy old data into the new chunk, adjusting for the new number of columns
                for row_index in 0..CHUNK_SIZE {
                    let old_start = row_index * (self.num_cols - 1);
                    let new_start = row_index * self.num_cols;
                    new_chunk[new_start..new_start + (self.num_cols - 1)]
                        .copy_from_slice(&old_chunk[old_start..old_start + (self.num_cols - 1)]);
                }

                // Replace the old chunk with the new chunk
                *chunk = (*self.arena).alloc(new_chunk) as *mut _;
            }
        }
    }
}

pub struct ArenaRowChunkedTableRawPointerDynamicArray {
    chunks: Vec<*mut DynamicArray>,  // Vec of raw pointers to DynamicArray for contiguous memory chunks
    num_rows: usize,
    num_cols: usize,                 // Dynamic number of columns in each row
    arena: *const Arena<f32>,        // Raw pointer to the arena
}

impl ArenaRowChunkedTableRawPointerDynamicArray {
    // Create a new table that references the arena for memory allocation
    pub fn new(arena: *const Arena<f32>, num_rows: usize, num_cols: usize) -> Self {
        let num_chunks = (num_rows + CHUNK_SIZE - 1) / CHUNK_SIZE;  // Calculate number of chunks needed
        let chunks = Vec::with_capacity(num_chunks);  // Start with an empty Vec for chunks

        ArenaRowChunkedTableRawPointerDynamicArray {
            chunks,
            num_rows,
            num_cols,
            arena,
        }
    }

    // Dynamically add a chunk if it doesn't exist
    fn add_chunk_if_needed(&mut self, chunk_index: usize) {
        if chunk_index >= self.chunks.len() {
            unsafe {
                // Allocate a new chunk using the arena (with raw pointer)
                let mut chunk = DynamicArray::new(self.arena, CHUNK_SIZE * self.num_cols);

                // Push the raw pointer of the chunk (DynamicArray) to self.chunks
                self.chunks.push(&mut chunk as *mut _);
            }
        }
    }
}

// Implementation of the IChunkedTable trait for ArenaRowChunkedTable
impl IChunkedTable for ArenaRowChunkedTableRawPointerDynamicArray {
    // Insert a value into the specific column index and row
    fn insert_value(&mut self, row: usize, col_idx: usize, value: f32) {
        let chunk_index = row / CHUNK_SIZE;
        let row_index = row % CHUNK_SIZE;

        // Ensure the chunk exists before inserting
        self.add_chunk_if_needed(chunk_index);

        unsafe {
            let chunk = self.chunks[chunk_index];
            let pos = row_index * self.num_cols + col_idx;  // Direct inline calculation
            (*chunk).set(pos, value);  // Set value in the DynamicArray
        }

        // Update the number of rows if the new row is beyond the current count
        if row >= self.num_rows {
            self.num_rows = row + 1;
        }
    }

    // Get a value from the specific column index and row
    fn get_value(&self, row: usize, col_idx: usize) -> f32 {
        let chunk_index = row / CHUNK_SIZE;
        let row_index = row % CHUNK_SIZE;

        unsafe {
            let chunk = self.chunks[chunk_index];
            let pos = row_index * self.num_cols + col_idx;  // Direct inline calculation
            (*chunk).get(pos)  // Get value from DynamicArray
        }
    }

    // Add a new column dynamically
    fn add_column(&mut self) {
        self.num_cols += 1;  // Increase the number of columns

        // Reallocate each chunk to accommodate the new column
        for chunk in &mut self.chunks {
            unsafe {
                let old_chunk = &mut **chunk;  // Dereference raw pointer to get the old chunk
                old_chunk.resize(CHUNK_SIZE * self.num_cols);  // Resize the DynamicArray
            }
        }
    }
}


struct ArenaRowChunkedTableSendRc {
    // Vec of Arc containing RefCell to allow thread-safe shared ownership and interior mutability
    chunks: Vec<SendRc<RefCell<[[f32; 3]; CHUNK_SIZE]>>>,
    num_rows: usize,
}

impl ArenaRowChunkedTableSendRc {
    // No lifetimes are needed since RefCell and Arc take care of interior mutability and thread safety
    fn new(arena: &Arena<[[f32; 3]; CHUNK_SIZE]>, num_rows: usize) -> Self {
        let num_chunks = (num_rows + CHUNK_SIZE - 1) / CHUNK_SIZE; // Calculate number of chunks needed

        let mut chunks = Vec::with_capacity(num_chunks);
        for _ in 0..num_chunks {
            // Allocate space for rows, each row has 3 columns and wrap it in RefCell and Arc for shared mutability
            chunks.push(SendRc::new(RefCell::new(*arena.alloc([[0.0; 3]; CHUNK_SIZE]))));
        }

        ArenaRowChunkedTableSendRc { chunks, num_rows }
    }

    // Insert a value into the chunked table using RefCell borrow_mut for mutation

}

impl IChunkedTable for ArenaRowChunkedTableSendRc {
    fn insert_value(&mut self, row: usize, col: usize, value: f32) {
        let chunk_index = row / CHUNK_SIZE;
        let row_index = row % CHUNK_SIZE;
        // Borrow the chunk mutably and modify the value
        self.chunks[chunk_index].borrow_mut()[row_index][col] = value;
    }

    // Get a value from the chunked table using RefCell borrow for immutably borrowing
    fn get_value(&self, row: usize, col: usize) -> f32 {
        let chunk_index = row / CHUNK_SIZE;
        let row_index = row % CHUNK_SIZE;
        // Borrow the chunk immutably and retrieve the value
        self.chunks[chunk_index].borrow()[row_index][col]
    }

    fn add_column(&mut self) {
        todo!()
    }
}


struct ArenaRowChunkedTableSoA {
    // Using IndexMap for dynamic column management, where each column is identified by index
    columns: IndexMap<usize, SendRc<RefCell<HashMap<usize,f32>>>>,
    num_rows: usize,
}

impl ArenaRowChunkedTableSoA {
    fn new(num_rows: usize) -> Self {
        let mut columns = IndexMap::new();

        // Initialize each column with space for `num_rows` elements
        columns.insert(
            0,
            SendRc::new(RefCell::new(Default::default())),
        );
        columns.insert(
            1,
            SendRc::new(RefCell::new(Default::default())),
        );
        columns.insert(
            2,
            SendRc::new(RefCell::new(Default::default())),
        );

        ArenaRowChunkedTableSoA { columns, num_rows }
    }
}

impl IChunkedTable for ArenaRowChunkedTableSoA {
    // Insert a value into the specific column index and row using RefCell borrow_mut for mutation
    fn insert_value(&mut self, row: usize, col_idx: usize, value: f32) {
        if let Some(column) = self.columns.get(&col_idx) {
            column.borrow_mut().insert(row, value);
        } else {
            panic!("Invalid column index: {}", col_idx);
        }
    }
    // Get a value from the specific column index and row using RefCell borrow for immutably borrowing
    fn get_value(&self, row: usize, col_idx: usize) -> f32 {
        if let Some(column) = self.columns.get(&col_idx) {
            column.borrow()[&row]
        } else {
            panic!("Invalid column index: {}", col_idx);
        }
    }
    // Add a new column dynamically (by index)
    fn add_column(&mut self) {
        let new_col_idx = self.columns.len(); // Assign a new column index
        self.columns.insert(
            new_col_idx,
            SendRc::new(RefCell::new(Default::default())),
        );
    }
}

trait IChunkedTable {
    // Insert a value into the specific column index and row using RefCell borrow_mut for mutation
    fn insert_value(&mut self, row: usize, col_idx: usize, value: f32);
    // Get a value from the specific column index and row using RefCell borrow for immutably borrowing
    fn get_value(&self, row: usize, col_idx: usize) -> f32;
    // Add a new column dynamically (by index)
    fn add_column(&mut self);
}


 fn test_harness<T : IChunkedTable, F: FnMut(usize, usize) -> T>(name: &str,mut table_factory: F) {
    // Define 100 million rows and 3 columns
    let num_rows: usize = 100_000_000;
    let num_cols: usize = 3;

    // Create an arena for allocating the table
    // Start timing for array creation
    let start = Instant::now();

    // Create the chunked table using arena allocation
    let mut table = table_factory(num_rows, num_cols);
    let mut rng = rand::thread_rng();

     // table.add_column();
     // table.add_column();
     println!("Populating table with random values");
    // Populate the table with random numbers
    for i in 0..num_rows {
        for j in 0..num_cols {
            let random_value: f32 = rng.gen();
            table.insert_value(i, j, random_value);
        }
    }

     let duration = start.elapsed();
     println!("Array creation (100 million by 3) using {} took: {:?}",name, duration);

     // println!("Verifying values");
     // for i in 0..num_rows {
     //     for j in 0..num_cols {
     //
     //         println!("Row: {}, Col: {}, Value: {}", i, j, table.get_value(i, j));
     //     }
     // }
     let start_add_column = Instant::now();
     table.add_column();
     // End timing for array creation
     let duration = start_add_column.elapsed();
     println!("Adding column took: {:?}", duration);
     // println!("Verifying values again");
     // for i in 0..num_rows {
     //     for j in 0..num_cols {
     //
     //         println!("Row: {}, Col: {}, Value: {}", i, j, table.get_value(i, j));
     //     }
     // }
     println!("Table populated with random values");

    // End timing for array creation

    // Set the threshold for filtering
    let threshold: f32 = 0.5;

    // Start timing for the filtering operation
    let start_filter = Instant::now();

    // Create a new table to store filtered results using the arena
    let mut filtered_table =table_factory(num_rows, num_cols);

    // Perform the filtering operation row by row
    for i in 0..num_rows {
        for j in 0..num_cols {
            let value = table.get_value(i, j);
            let filtered_value = if value > threshold { value } else { 0.0 };
            filtered_table.insert_value(i, j, filtered_value);
        }
    }

    // End timing for the filtering operation
    let duration_filter = start_filter.elapsed();
    println!(
        "Filtering operation (values > {}) using chunks and arenas took: {:?}",
        threshold, duration_filter
    );

    // Ensure some result is printed to avoid compiler optimizations
    println!("Filtered value at (100,000, 0): {}", filtered_table.get_value(0, 0));
}

fn flat_array_main() {
    let num_rows: usize = 100_000_000;
    let num_cols: usize = 3;

    let start = Instant::now();

    let mut a: Vec<f32> = vec![0.0; num_rows * num_cols];
    let mut rng = rand::thread_rng();

    for i in 0..(num_rows * num_cols) {
        a[i] = rng.gen::<f32>();
    }
    let duration = start.elapsed();
    println!("Array creation (100 million by 3) took: {:?}", duration);
    wait_for_key();

    filter_flat_array(num_rows, num_cols, &mut a);
}

fn filter_flat_array(num_rows: usize, num_cols: usize, a: &mut Vec<f32>) {
    let threshold: f32 = 0.5;
    let start_filter = Instant::now();

    let mut filtered_array: Vec<f32> = vec![0.0; num_rows * num_cols];
    let mut no_matches = 0;

    for i in 0..(num_rows * num_cols) {
        filtered_array[i] = if a[i] > threshold {
            a[i]
        } else {
            0.0
        };
    }

    let duration_filter = start_filter.elapsed();
    println!("Filtering operation (values > {}) took: {:?} - Matches: {}", threshold, duration_filter, filtered_array[100000]);
}

fn nested_array_main() {
    let num_rows: usize = 100_000_000;
    let num_cols: usize = 3;

    let start = Instant::now();

    let mut a: Vec<Vec<f32>> = vec![vec![0.0; num_cols]; num_rows];
    let mut rng = rand::thread_rng();

    for i in 0..num_rows {
        for j in 0..num_cols {
            a[i][j] = rng.gen::<f32>();
        }
    }

    let duration = start.elapsed();
    println!("Array creation (10 million by 3) took: {:?}", duration);

    wait_for_key();


    filter_nested_array(num_rows, num_cols, &mut a);
}

fn filter_nested_array(num_rows: usize, num_cols: usize, a: &mut Vec<Vec<f32>>) {
    let threshold: f32 = 0.5;
    let start_filter = Instant::now();

    let mut filtered_array: Vec<Vec<f32>> = vec![vec![0.0; num_cols]; num_rows];

    for j in 0..num_cols {
        for i in 0..num_rows {
            filtered_array[i][j] = if a[i][j] > threshold { a[i][j] } else { 0.0 };
        }
    }

    let duration_filter = start_filter.elapsed();
    println!("Filtering operation (values > {}) took: {:?}", threshold, duration_filter);
}

fn gpu_main() {
    set_device(0);
    info();
    println!("Arrayfire version: {:?}", get_version());
    let (name, platform, toolkit, compute) = device_info();
    println!("Device Info:\nName: {}\nPlatform: {}\nToolkit: {}\nCompute: {}", name, platform, toolkit, compute);

    let num_rows: u64 = 100_000_000;
    let num_cols: u64 = 3;

    let start = Instant::now();

    let dims = Dim4::new(&[num_rows, num_cols, 1, 1]);
    let mut a = randu::<f32>(dims);
    let threshold: f32 = 0.5;

    let duration = start.elapsed();
    println!("Array creation (100 million by 3) took: {:?}", duration);
    wait_for_key();

    filter_gpu(dims, a, threshold);
}

fn main_copy_data() {
    set_device(0);
    info();

    let original_rows = 100_000_000;
    let original_cols = 3;

    let new_rows = 50_000_000;

    // Creating a large initial array
    let dims = Dim4::new(&[original_rows, original_cols, 1, 1]);
    let large_array = randu::<f32>(dims);

    // Starting the timer
    let start = Instant::now();

    // Copying data from large_array to a new array with different dimensions
    let indices = Seq::new(0.0, new_rows as f64 - 1.0, 1.0);
    let new_array = index(&large_array, &[indices, Seq::default(), Seq::default(), Seq::default()]);

    // Stopping the timer
    let duration = start.elapsed();

    // Output the time taken for the copying operation
    println!("Time taken to copy array: {:?}", duration);
}

fn main_copy_data2() {
    set_device(0);
    info();

    let original_rows = 100_000_000;
    let original_cols = 3;

    // Creating a large initial array
    let dims = Dim4::new(&[original_rows, original_cols, 1, 1]);
    let large_array = randu::<f32>(dims);

    // Generating scattered indices for a non-contiguous data copy
    let mut scattered_indices = Vec::new();
    let mut rng = rand::thread_rng();
    let row_step = original_rows as usize / 50_000_000usize; // Adjust step to get approximately 50 million scattered rows
    for i in (0..original_rows).step_by(row_step) {
        scattered_indices.push(i);
    }

    // Converting vector of indices to ArrayFire Array
    let indices_array = Array::new(&scattered_indices, Dim4::new(&[scattered_indices.len() as u64, 1, 1, 1]));

    // Starting the timer
    let start = Instant::now();

    // Copying data from scattered memory locations to a new array
    let new_array = lookup(&large_array, &indices_array, 0);

    // Stopping the timer
    let duration = start.elapsed();

    // Output the time taken for the copying operation
    println!("Time taken to copy scattered array: {:?}", duration);
}



pub struct ArenaStaticArrayRowChunkedTable {
    chunks: Vec<*mut f32>,  // Raw pointers to chunks of f32 (allocated in the arena)
    num_rows: usize,
    num_cols: usize,        // Dynamic number of columns per row
    arena: Arena<f32>,  // Arena for memory allocation
}

impl ArenaStaticArrayRowChunkedTable {
    // Create a new table with the given number of rows and columns
    pub fn new(arena: Arena<f32>, num_rows: usize, num_cols: usize) -> Self {
        let num_chunks = (num_rows + CHUNK_SIZE - 1) / CHUNK_SIZE; // Calculate number of chunks needed
        let mut chunks = Vec::with_capacity(num_chunks);           // Preallocate the chunk array

        let mut table = ArenaStaticArrayRowChunkedTable {
            chunks,
            num_rows,
            num_cols,
            arena,
        };

        for _ in 0..num_chunks {
            // Allocate memory for each chunk (CHUNK_SIZE rows * num_cols columns)
            let chunk = unsafe { table.arena.alloc_uninitialized(CHUNK_SIZE * num_cols).as_mut_ptr()  as *mut f32 };
            table.chunks.push(chunk);
        }

        table
    }

    // Dynamically adds a chunk if needed
    fn add_chunk_if_needed(&mut self) {
        if self.num_rows % CHUNK_SIZE == 0 {
            let chunk = unsafe { self.arena.alloc_uninitialized(CHUNK_SIZE * self.num_cols).as_mut_ptr() as *mut f32  };
            self.chunks.push(chunk);
        }
    }

    // Create a new arena and copy all data to the new arena with an additional column
    pub fn add_column_with_new_arena(&mut self, new_arena: Arena<f32>) {
        let new_num_cols = self.num_cols + 1;

        // Create a new vector for the new arena chunks
        let mut new_chunks = Vec::with_capacity(self.chunks.len());

        // Allocate new chunks in the new arena with the additional column
        for &old_chunk in &self.chunks {
            let new_chunk = unsafe { new_arena.alloc_uninitialized(CHUNK_SIZE * new_num_cols).as_mut_ptr()  as *mut f32 };

            // Copy the old data to the new chunk
            for row_index in 0..CHUNK_SIZE {
                let old_pos = row_index * self.num_cols;
                let new_pos = row_index * new_num_cols;
                unsafe {
                    // Copy the old row data (excluding the new column) to the new chunk
                    std::ptr::copy_nonoverlapping(old_chunk.add(old_pos), new_chunk.add(new_pos), self.num_cols);
                }
            }

            new_chunks.push(new_chunk);
        }

        // Replace the old chunks with the new ones
        self.chunks = new_chunks;
        self.num_cols = new_num_cols; // Update the column count
        self.arena = new_arena;       // Switch to the new arena

        // The old arena will be dropped once it goes out of scope, freeing the old memory
    }

}

// Implementation of the IChunkedTable trait
impl IChunkedTable for ArenaStaticArrayRowChunkedTable {
    // Insert a value into the chunked table
    fn insert_value(&mut self, row: usize, col: usize, value: f32) {
        if row >= self.num_rows {
            self.add_chunk_if_needed();
            self.num_rows += 1;
        }
        let chunk_index = row / CHUNK_SIZE;
        let row_index = row % CHUNK_SIZE;
        let pos = row_index * self.num_cols + col;  // Calculate position in the flat array
        unsafe {
            *self.chunks[chunk_index].add(pos) = value;
        }
    }

    // Get a value from the chunked table
    fn get_value(&self, row: usize, col: usize) -> f32 {
        let chunk_index = row / CHUNK_SIZE;
        let row_index = row % CHUNK_SIZE;
        let pos = row_index * self.num_cols + col;  // Calculate position in the flat array
        unsafe {
            *self.chunks[chunk_index].add(pos)
        }
    }

    // Add a new column dynamically (resizing each chunk to fit the new column)
    fn add_column(&mut self) {
        self.add_column_with_new_arena(Arena::new());
    }
}

// fn main_hash_map() {
//     set_device(0);
//     info();
//
//     let original_rows = 100_000_000usize;
//     let original_cols = 3usize;
//
//     // Creating a sparse array simulation with HashMap
//     let mut sparse_map: HashMap<u64, f64> = HashMap::new();
//     let mut rng: ThreadRng = rand::thread_rng();
//     let index_dist = Uniform::new(0, original_rows as u64);
//
//     // Populate the HashMap with random data
//     for _ in 0..50_000_000 {
//         let index = rng.sample(&index_dist);
//         let value = rng.gen::<f64>();  // Generate random f64 values
//         sparse_map.insert(index, value);
//     }
//
//     // Extract indices and values from the map
//     let mut indices: Vec<u64> = Vec::new();
//     let mut values: Vec<f64> = Vec::new();
//     for (index, value) in sparse_map {
//         indices.push(index);
//         values.push(value);
//     }
//
//     // Creating ArrayFire arrays from the indices and values
//     let af_indices = Array::new(&indices, Dim4::new(&[indices.len() as u64, 1, 1, 1]));
//     let af_values = Array::new(&values, Dim4::new(&[values.len() as u64, 1, 1, 1]));
//
//     // Creating a dense array to fill with the sparse data
//     let large_array_dims = Dim4::new(&[original_rows as u64, original_cols as u64, 1, 1]);
//     let mut large_array = constant(0.0f64, large_array_dims);
//
//     // Starting the timer
//     let start = Instant::now();
//
//     // Efficiently inserting values into the large array using gfor (generalized for-loop)
//     gfor(seq!(0u64, indices.len() as u64 - 1, 1), |i| {
//         let idx = af_indices.get(&seq!(i));
//         let val = af_values.get(&seq!(i));
//         large_array.assign_seq(&idx, &val);
//     });
//
//     // Stopping the timer
//     let duration = start.elapsed();
//
//     // Output the time taken for the operation
//     println!("Time taken to populate large array with sparse data: {:?}", duration);
// }
fn filter_gpu(dims: Dim4, a: Array<f32>, threshold: f32) {
    let start_filter = Instant::now();

    let filter_mask = gt(&a, &threshold, false);
    let filtered_array = select(&a, &filter_mask, &constant(0.0f32, dims));

    let duration_filter = start_filter.elapsed();
    println!("Filtering operation (values > {}) took: {:?} - {}", threshold, duration_filter,filtered_array.dims().index(3usize));
}


fn wait_for_key() {
    println!("Press any key to continue...");
    let _ = std::io::stdin().read(&mut [0u8]).unwrap();
}