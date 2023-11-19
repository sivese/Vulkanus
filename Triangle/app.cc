#include <vulkan/vulkan.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#include <iostream>
#include <vector>
#include <map>
#include <optional>

struct QueueFamilyIndices {
	uint32_t graphicsFamily;
};

class Application {
public:
	auto run() -> void {
		this->init();

		this->mainLoop();
	}

private:
	SDL_Window* window = nullptr;

	VkInstance instance = VK_NULL_HANDLE;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	
	//std::string errorMessage;
	std::string appName = "Vulkanus";
	std::string engineName = "V-TWIN";

	bool quit = false;

	/* 
			initialization and create vulkan instance
	*/
	auto init() -> void {
		if (SDL_Init(SDL_INIT_VIDEO) < 0)
			throw std::runtime_error(SDL_GetError());

		this->window = SDL_CreateWindow(
			appName.c_str(),				// title
			SDL_WINDOWPOS_CENTERED, // x position
			SDL_WINDOWPOS_CENTERED, // y position
			600, 600,				// width, height value
			SDL_WINDOW_SHOWN | SDL_WINDOW_VULKAN	// window flag
		);

		createInstance();
	}

	auto createInstance() -> void {
		auto sdlExtensionCount = uint32_t(0);
		SDL_Vulkan_GetInstanceExtensions(this->window, &sdlExtensionCount, nullptr);

		auto extensionNames = std::vector<const char*>(sdlExtensionCount);
		SDL_Vulkan_GetInstanceExtensions(this->window, &sdlExtensionCount, extensionNames.data());

		auto appInfo = VkApplicationInfo {
			.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
			.pApplicationName = appName.c_str(),
			.applicationVersion = VK_MAKE_VERSION(0, 0, 1),
			.pEngineName = engineName.c_str(),
			.engineVersion = VK_MAKE_VERSION(0, 0, 1),
			.apiVersion = VK_API_VERSION_1_3
		};

		auto createInfo = VkInstanceCreateInfo {
			.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
			.pApplicationInfo = &appInfo,
			.enabledLayerCount = 0,		// validation layer size
			.ppEnabledLayerNames = nullptr,
			.enabledExtensionCount = sdlExtensionCount,
			.ppEnabledExtensionNames = extensionNames.data()
		};

		auto result = vkCreateInstance(&createInfo, nullptr, &(this->instance));

		if (result != VK_SUCCESS)
			throw std::runtime_error("Failed to create vulkan instance");

		auto extensionCount = uint32_t(0);
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

		auto extensions = std::vector<VkExtensionProperties>(extensionCount);

		std::cout << "available extensions:\n";

		for (const auto& extension : extensions)
			std::cout << '\t' << extension.extensionName << '\n';
	}

	/*
		Physical device methods
	*/
	auto pickPhysicalDevice() {
		auto deviceCount = uint32_t(0);
		vkEnumeratePhysicalDevices(this->instance, &deviceCount, nullptr);

		if (deviceCount == 0)
			throw std::runtime_error("failed to find GPUs with Vulkan support!");

		auto devices = std::vector<VkPhysicalDevice>(deviceCount);
		vkEnumeratePhysicalDevices(this->instance, &deviceCount, devices.data());

		for (const auto& device : devices) {
			if (isDeviceSuitable(device)) {
				this->physicalDevice = device;
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE) {
			throw std::runtime_error("failed to find a suitable GPU!");
		}

		//auto candidates = std::multimap<int, VkPhysicalDevice>();
	}

	auto isDeviceSuitable(VkPhysicalDevice device) -> bool {
		auto deviceProperties = VkPhysicalDeviceProperties();
		vkGetPhysicalDeviceProperties(device, &deviceProperties);

		auto deviceFeatures = VkPhysicalDeviceFeatures();
		vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

		return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU 
			&& deviceFeatures.geometryShader;
	}

	auto rateDeviceSuitability(VkPhysicalDevice device) -> int {
		int score = 0;
		
		return score;
	}


	/*
	
	*/
	auto findQueueFamilies(VkPhysicalDevice deivce) -> uint32_t {
		auto graphicsFamily = std::optional<uint32_t>();
		std::cout << std::boolalpha << graphicsFamily.has_value() << std::endl;

		graphicsFamily = 0;
		std::cout<<std::boolalpha
	}

	auto mainLoop() -> void {
		auto event = SDL_Event();

		while (quit == false) {
			while (SDL_PollEvent(&event)) {
				switch (event.type) {
				case SDL_QUIT:
					this->quit = true;
					break;
				}
			}
		}
	}

	/*
			The last step, destroy all thing
	*/

	auto cleanUp() -> void {
		vkDestroyInstance(this->instance, nullptr);

		SDL_DestroyWindow(this->window);
		SDL_Vulkan_UnloadLibrary();
		SDL_Quit();
	}
};

auto main(int argc, char* argv[]) -> int {
	auto app = Application();

	try {
		app.run();
	}
	catch (std::exception exp) {
		std::cout << exp.what() << std::endl;
	}

	return 0;
}