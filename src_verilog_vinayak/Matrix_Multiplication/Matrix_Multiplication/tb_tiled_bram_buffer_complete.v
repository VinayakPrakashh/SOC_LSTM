`timescale 1ns / 1ps

//==============================================================================
// Complete Testbench for Tiled BRAM Buffer (LUTRAM)
// Author: Generated for SOC_LSTM project
// Date: October 6, 2025
// Description: Comprehensive testbench covering all aspects of LUTRAM functionality
//==============================================================================

module tb_tiled_bram_buffer_complete();

    //==========================================================================
    // Parameters and Local Parameters
    //==========================================================================
    parameter DATA_WIDTH = 12;
    parameter TILE_SIZE  = 16;
    localparam ADDR_WIDTH = $clog2(TILE_SIZE);
    localparam MAX_DATA_VALUE = (1 << DATA_WIDTH) - 1;
    
    //==========================================================================
    // Clock and Reset Generation
    //==========================================================================
    reg clk = 1'b0;
    reg rst = 1'b0;
    
    // Clock generation - 100MHz
    always #5 clk = ~clk;
    
    //==========================================================================
    // DUT Interface Signals
    //==========================================================================
    reg                    wr_en = 1'b0;
    reg [ADDR_WIDTH-1:0]   addr = 0;
    reg [DATA_WIDTH-1:0]   data_in = 0;
    wire [DATA_WIDTH-1:0]  data_out;
    
    //==========================================================================
    // Test Control and Status Variables
    //==========================================================================
    integer test_num = 0;
    integer pass_count = 0;
    integer fail_count = 0;
    integer total_tests = 0;
    
    // Test patterns and expected results
    reg [DATA_WIDTH-1:0] reference_memory [0:TILE_SIZE-1];
    reg [DATA_WIDTH-1:0] test_data;
    reg [ADDR_WIDTH-1:0] test_addr;
    
    // Test control flags
    reg test_running = 1'b0;
    reg verbose_mode = 1'b1;
    
    //==========================================================================
    // Device Under Test (DUT) Instantiation
    //==========================================================================
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
    
    //==========================================================================
    // Test Monitoring and Logging
    //==========================================================================
    
    // Transaction monitor
    always @(posedge clk) begin
        if (test_running && verbose_mode) begin
            if (wr_en) begin
                $display("[%0t] WRITE: Addr=0x%h (%0d), Data=0x%h (%0d)", 
                         $time, addr, addr, data_in, data_in);
            end else begin
                $display("[%0t] READ:  Addr=0x%h (%0d), Data=0x%h (%0d)", 
                         $time, addr, addr, data_out, data_out);
            end
        end
    end
    
    //==========================================================================
    // Utility Tasks
    //==========================================================================
    
    // Reset sequence
    task apply_reset();
        begin
            $display("\n[INFO] Applying reset sequence...");
            rst = 1'b1;
            wr_en = 1'b0;
            addr = 0;
            data_in = 0;
            repeat(5) @(posedge clk);
            rst = 1'b0;
            repeat(3) @(posedge clk);
            $display("[INFO] Reset sequence completed");
        end
    endtask
    
    // Write to memory
    task write_memory(
        input [ADDR_WIDTH-1:0] write_addr,
        input [DATA_WIDTH-1:0] write_data
    );
        begin
            @(posedge clk);
            wr_en = 1'b1;
            addr = write_addr;
            data_in = write_data;
            reference_memory[write_addr] = write_data;
            @(posedge clk);
            wr_en = 1'b0;
        end
    endtask
    
    // Read from memory
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
    
    // Verify read data
    task verify_read(
        input [ADDR_WIDTH-1:0] check_addr,
        input [DATA_WIDTH-1:0] expected_data,
        input [255:0] test_description
    );
        reg [DATA_WIDTH-1:0] actual_data;
        begin
            read_memory(check_addr, actual_data);
            total_tests = total_tests + 1;
            
            if (actual_data === expected_data) begin
                pass_count = pass_count + 1;
                if (verbose_mode) begin
                    $display("✓ PASS: %s - Addr=%0d, Expected=0x%h, Got=0x%h", 
                             test_description, check_addr, expected_data, actual_data);
                end
            end else begin
                fail_count = fail_count + 1;
                $display("✗ FAIL: %s - Addr=%0d, Expected=0x%h, Got=0x%h", 
                         test_description, check_addr, expected_data, actual_data);
            end
        end
    endtask
    
    // Initialize reference memory
    task init_reference_memory();
        integer i;
        begin
            for (i = 0; i < TILE_SIZE; i = i + 1) begin
                reference_memory[i] = 0;
            end
        end
    endtask
    
    // Display test header
    task display_test_header(input [255:0] test_name);
        begin
            $display("\n" + "="*60);
            $display("TEST %0d: %s", test_num, test_name);
            $display("="*60);
            test_num = test_num + 1;
        end
    endtask
    
    //==========================================================================
    // Test Cases
    //==========================================================================
    
    // Test 1: Basic Write and Read Operations
    task test_basic_write_read();
        integer i;
        begin
            display_test_header("Basic Write/Read Operations");
            
            // Test each memory location with simple patterns
            for (i = 0; i < TILE_SIZE; i = i + 1) begin
                test_data = i * 17 + 5; // Simple pattern
                write_memory(i, test_data);
                verify_read(i, test_data, $sformatf("Basic W/R Addr %0d", i));
            end
        end
    endtask
    
    // Test 2: Data Pattern Tests
    task test_data_patterns();
        begin
            display_test_header("Data Pattern Tests");
            
            // All zeros
            write_memory(0, 12'h000);
            verify_read(0, 12'h000, "All Zeros Pattern");
            
            // All ones
            write_memory(1, MAX_DATA_VALUE);
            verify_read(1, MAX_DATA_VALUE, "All Ones Pattern");
            
            // Alternating patterns
            write_memory(2, 12'hAAA);
            verify_read(2, 12'hAAA, "Alternating 1010 Pattern");
            
            write_memory(3, 12'h555);
            verify_read(3, 12'h555, "Alternating 0101 Pattern");
            
            // Walking ones
            write_memory(4, 12'h001);
            verify_read(4, 12'h001, "Walking Ones - Bit 0");
            
            write_memory(5, 12'h002);
            verify_read(5, 12'h002, "Walking Ones - Bit 1");
            
            write_memory(6, 12'h004);
            verify_read(6, 12'h004, "Walking Ones - Bit 2");
            
            write_memory(7, 12'h800);
            verify_read(7, 12'h800, "Walking Ones - MSB");
        end
    endtask
    
    // Test 3: Address Boundary Testing
    task test_address_boundaries();
        begin
            display_test_header("Address Boundary Tests");
            
            // Minimum address
            write_memory(0, 12'h123);
            verify_read(0, 12'h123, "Minimum Address (0)");
            
            // Maximum address
            write_memory(TILE_SIZE-1, 12'h456);
            verify_read(TILE_SIZE-1, 12'h456, "Maximum Address");
            
            // Middle addresses
            write_memory(TILE_SIZE/2, 12'h789);
            verify_read(TILE_SIZE/2, 12'h789, "Middle Address");
            
            write_memory(TILE_SIZE/4, 12'hABC);
            verify_read(TILE_SIZE/4, 12'hABC, "Quarter Address");
            
            write_memory(3*TILE_SIZE/4, 12'hDEF);
            verify_read(3*TILE_SIZE/4, 12'hDEF, "Three-Quarter Address");
        end
    endtask
    
    // Test 4: Random Access Patterns
    task test_random_access();
        integer i, rand_addr, rand_data;
        integer num_random_tests = 50;
        begin
            display_test_header("Random Access Pattern Tests");
            
            for (i = 0; i < num_random_tests; i = i + 1) begin
                rand_addr = $urandom % TILE_SIZE;
                rand_data = $urandom % (1 << DATA_WIDTH);
                
                write_memory(rand_addr, rand_data);
                verify_read(rand_addr, rand_data, 
                           $sformatf("Random Test %0d", i+1));
                
                // Small delay between operations
                repeat(2) @(posedge clk);
            end
        end
    endtask
    
    // Test 5: Overwrite Testing
    task test_overwrite_operations();
        integer i;
        begin
            display_test_header("Overwrite Operations Test");
            
            // Fill memory with initial pattern
            for (i = 0; i < TILE_SIZE; i = i + 1) begin
                write_memory(i, i * 11);
            end
            
            // Verify initial pattern
            for (i = 0; i < TILE_SIZE; i = i + 1) begin
                verify_read(i, i * 11, 
                           $sformatf("Initial Pattern Addr %0d", i));
            end
            
            // Overwrite with new pattern
            for (i = 0; i < TILE_SIZE; i = i + 1) begin
                write_memory(i, (TILE_SIZE-1-i) * 13);
            end
            
            // Verify overwritten pattern
            for (i = 0; i < TILE_SIZE; i = i + 1) begin
                verify_read(i, (TILE_SIZE-1-i) * 13, 
                           $sformatf("Overwrite Pattern Addr %0d", i));
            end
        end
    endtask
    
    // Test 6: Simultaneous Read/Write Testing
    task test_simultaneous_read_write();
        reg [DATA_WIDTH-1:0] read_result;
        begin
            display_test_header("Simultaneous Read/Write Test");
            
            // Pre-load some data
            write_memory(5, 12'h111);
            
            // Test: Write new data while reading from same address
            @(posedge clk);
            wr_en = 1'b1;
            addr = 5;
            data_in = 12'h222;
            
            @(posedge clk);
            read_result = data_out;
            wr_en = 1'b0;
            
            // The read should return the old value during write
            if (read_result === 12'h111) begin
                pass_count = pass_count + 1;
                $display("✓ PASS: Simultaneous R/W - Read old value during write");
            end else begin
                fail_count = fail_count + 1;
                $display("✗ FAIL: Simultaneous R/W - Expected 0x111, Got 0x%h", read_result);
            end
            total_tests = total_tests + 1;
            
            // Verify new value was written
            verify_read(5, 12'h222, "Simultaneous R/W - New value written");
        end
    endtask
    
    // Test 7: Memory Independence Test
    task test_memory_independence();
        integer i, j;
        begin
            display_test_header("Memory Location Independence Test");
            
            // Write unique pattern to each location
            for (i = 0; i < TILE_SIZE; i = i + 1) begin
                write_memory(i, i * i + i + 1); // Unique pattern for each address
            end
            
            // Verify each location independently
            for (i = 0; i < TILE_SIZE; i = i + 1) begin
                verify_read(i, i * i + i + 1, 
                           $sformatf("Independence Test Addr %0d", i));
            end
            
            // Modify one location and verify others unchanged
            write_memory(TILE_SIZE/2, 12'hFFF);
            
            for (i = 0; i < TILE_SIZE; i = i + 1) begin
                if (i == TILE_SIZE/2) begin
                    verify_read(i, 12'hFFF, "Modified Location");
                end else begin
                    verify_read(i, i * i + i + 1, 
                               $sformatf("Unchanged Location %0d", i));
                end
            end
        end
    endtask
    
    // Test 8: Stress Test
    task test_stress_operations();
        integer i, stress_cycles = 200;
        integer rand_addr, rand_data;
        begin
            display_test_header("Stress Test - Rapid Operations");
            
            for (i = 0; i < stress_cycles; i = i + 1) begin
                rand_addr = $urandom % TILE_SIZE;
                rand_data = $urandom % (1 << DATA_WIDTH);
                
                // Rapid write followed by read
                write_memory(rand_addr, rand_data);
                verify_read(rand_addr, rand_data, 
                           $sformatf("Stress Test Cycle %0d", i+1));
            end
        end
    endtask
    
    //==========================================================================
    // Main Test Sequence
    //==========================================================================
    initial begin
        // Test environment setup
        $display("\n" + "="*80);
        $display("COMPLETE TESTBENCH FOR TILED BRAM BUFFER (LUTRAM)");
        $display("="*80);
        $display("Configuration:");
        $display("  - Data Width: %0d bits", DATA_WIDTH);
        $display("  - Tile Size: %0d entries", TILE_SIZE);
        $display("  - Address Width: %0d bits", ADDR_WIDTH);
        $display("  - Max Data Value: 0x%h (%0d)", MAX_DATA_VALUE, MAX_DATA_VALUE);
        $display("="*80);
        
        // Initialize test environment
        test_running = 1'b1;
        init_reference_memory();
        apply_reset();
        
        // Execute all test cases
        test_basic_write_read();
        test_data_patterns();
        test_address_boundaries();
        test_random_access();
        test_overwrite_operations();
        test_simultaneous_read_write();
        test_memory_independence();
        
        // Disable verbose mode for stress test
        verbose_mode = 1'b0;
        test_stress_operations();
        verbose_mode = 1'b1;
        
        // Test completion
        test_running = 1'b0;
        
        // Final results
        $display("\n" + "="*80);
        $display("FINAL TEST RESULTS");
        $display("="*80);
        $display("Total Tests Executed: %0d", total_tests);
        $display("Tests Passed: %0d", pass_count);
        $display("Tests Failed: %0d", fail_count);
        $display("Success Rate: %0.1f%%", (pass_count * 100.0) / total_tests);
        
        if (fail_count == 0) begin
            $display("\n🎉 ALL TESTS PASSED! LUTRAM MODULE IS WORKING CORRECTLY! 🎉");
        end else begin
            $display("\n❌ %0d TESTS FAILED - PLEASE REVIEW THE FAILURES", fail_count);
        end
        
        $display("="*80);
        $display("Testbench completed at time: %0t", $time);
        $display("="*80);
        
        $stop;
    end
    
    //==========================================================================
    // Safety and Monitoring
    //==========================================================================
    
    // Timeout protection
    initial begin
        #100000; // 100 microseconds
        $display("\n❌ ERROR: Testbench timeout after 100us!");
        $display("This may indicate a problem with the DUT or testbench.");
        $stop;
    end
    
    // Monitor for X or Z values
    always @(posedge clk) begin
        if (test_running) begin
            if (^data_out === 1'bx) begin
                $display("⚠️  WARNING: Unknown (X) value detected on data_out at time %0t", $time);
            end
            if (^data_out === 1'bz) begin
                $display("⚠️  WARNING: High-impedance (Z) value detected on data_out at time %0t", $time);
            end
        end
    end

endmodule