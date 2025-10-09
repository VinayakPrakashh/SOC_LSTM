`timescale 1ns / 1ps

module tb_tiled_bram_buffer();

    // Parameters - matching the DUT
    parameter DATA_WIDTH = 12;
    parameter TILE_SIZE  = 16;
    parameter ADDR_WIDTH = $clog2(TILE_SIZE);
    
    // Clock and Reset
    reg clk = 1'b0;
    reg rst = 1'b0;
    
    // Input signals
    reg wr_en = 1'b0;
    reg [ADDR_WIDTH-1:0] addr = 0;
    reg [DATA_WIDTH-1:0] data_in = 0;
    
    // Output signals
    wire [DATA_WIDTH-1:0] data_out;
    
    // Test variables
    integer i;
    integer errors = 0;
    integer test_count = 0;
    reg [DATA_WIDTH-1:0] expected_data;
    reg [DATA_WIDTH-1:0] test_pattern [0:TILE_SIZE-1];
    
    // DUT instantiation
    tiled_bram_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_SIZE(TILE_SIZE)
    ) dut (
        .clk(clk),
        .rst(rst),
        .wr_en(wr_en),
        .addr(addr),
        .data_in(data_in),
        .data_out(data_out)
    );
    
    // Clock generation
    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;  // 100MHz clock (10ns period)
    end
    
    // Test stimulus
    initial begin
        $display("========================================");
        $display("Starting LUTRAM Buffer Testbench");
        $display("Data Width: %0d bits", DATA_WIDTH);
        $display("Tile Size: %0d entries", TILE_SIZE);
        $display("Address Width: %0d bits", ADDR_WIDTH);
        $display("========================================");
        
        // Initialize
        initialize_signals();
        
        // Test 1: Basic Write/Read Test
        test_basic_write_read();
        
        // Test 2: Sequential Write/Read Test
        test_sequential_write_read();
        
        // Test 3: Random Access Test
        test_random_access();
        
        // Test 4: Simultaneous Read/Write Test
        test_simultaneous_read_write();
        
        // Test 5: Address Boundary Test
        test_address_boundaries();
        
        // Test 6: Data Pattern Test
        test_data_patterns();
        
        // Final Results
        display_test_results();
        
        $stop;
    end
    
    // Task to initialize signals
    task initialize_signals;
        begin
            $display("\n=== Initializing Signals ===");
            rst = 1'b1;
            wr_en = 1'b0;
            addr = 0;
            data_in = 0;
            
            repeat(3) @(posedge clk);
            rst = 1'b0;
            repeat(2) @(posedge clk);
            
            $display("Reset sequence completed");
        end
    endtask
    
    // Task to write data to memory
    task write_memory(
        input [ADDR_WIDTH-1:0] write_addr,
        input [DATA_WIDTH-1:0] write_data
    );
        begin
            @(posedge clk);
            wr_en = 1'b1;
            addr = write_addr;
            data_in = write_data;
            @(posedge clk);
            wr_en = 1'b0;
        end
    endtask
    
    // Task to read data from memory
    task read_memory(
        input [ADDR_WIDTH-1:0] read_addr,
        output [DATA_WIDTH-1:0] read_data
    );
        begin
            @(posedge clk);
            wr_en = 1'b0;
            addr = read_addr;
            @(posedge clk);
            read_data = data_out;
        end
    endtask
    
    // Task to check read data
    task check_data(
        input [ADDR_WIDTH-1:0] check_addr,
        input [DATA_WIDTH-1:0] expected,
        input [7*8:1] test_name
    );
        reg [DATA_WIDTH-1:0] actual;
        begin
            read_memory(check_addr, actual);
            if (actual === expected) begin
                $display("✓ PASS [%s]: Addr=%0d, Expected=%0d, Got=%0d", 
                         test_name, check_addr, expected, actual);
            end else begin
                $display("✗ FAIL [%s]: Addr=%0d, Expected=%0d, Got=%0d", 
                         test_name, check_addr, expected, actual);
                errors = errors + 1;
            end
            test_count = test_count + 1;
        end
    endtask
    
    // Test 1: Basic Write/Read
    task test_basic_write_read;
        begin
            $display("\n=== Test 1: Basic Write/Read ===");
            
            // Write a simple value
            write_memory(4'd5, 12'd100);
            
            // Read it back
            check_data(4'd5, 12'd100, "Basic");
            
            // Write different value to same location
            write_memory(4'd5, 12'd200);
            check_data(4'd5, 12'd200, "Overwrite");
        end
    endtask
    
    // Test 2: Sequential Write/Read
    task test_sequential_write_read;
        begin
            $display("\n=== Test 2: Sequential Write/Read ===");
            
            // Write sequential pattern
            for (i = 0; i < TILE_SIZE; i = i + 1) begin
                write_memory(i, i * 10 + 5);  // Pattern: 5, 15, 25, 35...
                test_pattern[i] = i * 10 + 5;
            end
            
            // Read back and verify
            for (i = 0; i < TILE_SIZE; i = i + 1) begin
                check_data(i, test_pattern[i], "Sequential");
            end
        end
    endtask
    
    // Test 3: Random Access
    task test_random_access;
        integer rand_addr, rand_data;
        reg [DATA_WIDTH-1:0] memory_model [0:TILE_SIZE-1];
        begin
            $display("\n=== Test 3: Random Access Test ===");
            
            // Random write pattern
            for (i = 0; i < TILE_SIZE * 2; i = i + 1) begin
                rand_addr = $urandom % TILE_SIZE;
                rand_data = $urandom % (1 << DATA_WIDTH);
                
                write_memory(rand_addr, rand_data);
                memory_model[rand_addr] = rand_data;
                
                // Immediate read-back check
                check_data(rand_addr, rand_data, "Random");
            end
        end
    endtask
    
    // Test 4: Simultaneous Read/Write (same cycle)
    task test_simultaneous_read_write;
        reg [DATA_WIDTH-1:0] read_result;
        begin
            $display("\n=== Test 4: Simultaneous Read/Write ===");
            
            // Pre-load some data
            write_memory(4'd3, 12'd777);
            
            // Test simultaneous read/write to same address
            @(posedge clk);
            wr_en = 1'b1;
            addr = 4'd3;
            data_in = 12'd888;
            
            @(posedge clk);
            read_result = data_out;
            wr_en = 1'b0;
            
            $display("Simultaneous R/W: Previous value read = %0d", read_result);
            
            // Verify new value was written
            check_data(4'd3, 12'd888, "SimRW");
        end
    endtask
    
    // Test 5: Address Boundaries
    task test_address_boundaries;
        begin
            $display("\n=== Test 5: Address Boundary Test ===");
            
            // Test minimum address
            write_memory(0, 12'd1111);
            check_data(0, 12'd1111, "MinAddr");
            
            // Test maximum address
            write_memory(TILE_SIZE-1, 12'd2222);
            check_data(TILE_SIZE-1, 12'd2222, "MaxAddr");
            
            // Test middle address
            write_memory(TILE_SIZE/2, 12'd3333);
            check_data(TILE_SIZE/2, 12'd3333, "MidAddr");
        end
    endtask
    
    // Test 6: Data Patterns
    task test_data_patterns;
        begin
            $display("\n=== Test 6: Data Pattern Test ===");
            
            // All zeros
            write_memory(4'd1, 12'd0);
            check_data(4'd1, 12'd0, "AllZeros");
            
            // All ones (within data width)
            write_memory(4'd2, (1 << DATA_WIDTH) - 1);
            check_data(4'd2, (1 << DATA_WIDTH) - 1, "AllOnes");
            
            // Alternating pattern
            write_memory(4'd3, 12'b101010101010);
            check_data(4'd3, 12'b101010101010, "Alt1010");
            
            write_memory(4'd4, 12'b010101010101);
            check_data(4'd4, 12'b010101010101, "Alt0101");
            
            // Walking ones
            for (i = 0; i < DATA_WIDTH && i < TILE_SIZE; i = i + 1) begin
                write_memory(i, 1 << i);
                check_data(i, 1 << i, "Walking1");
            end
        end
    endtask
    
    // Task to display test results
    task display_test_results;
        begin
            $display("\n========================================");
            $display("LUTRAM Buffer Test Results");
            $display("========================================");
            $display("Total Tests: %0d", test_count);
            $display("Passed: %0d", test_count - errors);
            $display("Failed: %0d", errors);
            
            if (errors == 0) begin
                $display("🎉 ALL TESTS PASSED! 🎉");
            end else begin
                $display("❌ %0d TESTS FAILED", errors);
            end
            
            $display("========================================");
        end
    endtask
    
    // Monitor for continuous observation
    always @(posedge clk) begin
        if (wr_en) begin
            $display("Time %0t: WRITE - Addr=%0d, Data=%0d", $time, addr, data_in);
        end else begin
            $display("Time %0t: READ  - Addr=%0d, Data=%0d", $time, addr, data_out);
        end
    end
    
    // Safety timeout
    initial begin
        #50000;  // 50 microseconds
        $display("ERROR: Testbench timeout!");
        $stop;
    end

endmodule