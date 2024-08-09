local lfs = require("lfs")

local function run_python_script()
    local handle, err = io.popen("python3 training_script_with_logs.py 2>&1")
    if not handle then
        print("Failed to run Python script: " .. (err or "unknown error"))
        return false, err, -1
    end

    local result, read_err = handle:read("*a")
    if not result then
        print("Failed to read script output: " .. (read_err or "unknown error"))
        handle:close()
        return false, read_err, -1
    end

    local success, _, exit_code = handle:close()
    -- Check for specific errors in the output
    if result:find("RuntimeError: ") then
        print("Runtime error detected in output")
        return false, result, exit_code or -1
    end
    if result:find("ValueError: ") then
        print("ValueError detected in output")
        return false, result, exit_code or -1
    end

    return success, result, exit_code
end

local max_attempts = 10
local attempt = 1

while attempt <= max_attempts do
    print("Attempt " .. attempt .. " of " .. max_attempts)
    local success, output, exit_code = run_python_script()
    if success then
        print("Python script output:")
        print(output)
        print("Python script ran successfully.")
        break
    else
        print("Error running Python script. Output:")
        print(output)
        print("Exit code:", exit_code)
        if attempt < max_attempts then
            print("Retrying in 5 seconds...")
            os.execute("sleep 5")
        else
            print("Max attempts reached. Script failed to run successfully.")
        end
    end
    attempt = attempt + 1
end

